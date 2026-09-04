//! JSON value codec, mirroring `runtime/c/codec.c` and `runtime/include/codec.h`.
//!
//! This is the *canonical* encoder used by the store to serialize values into
//! the journal and snapshots. It differs from `data::json_stringify` in two
//! deliberate ways, both preserved here for byte-identical output:
//!
//! * control characters `\b \f \n \r \t` are emitted as named escapes, and
//!   other bytes below `0x20` as `\u00xx`;
//! * numbers use `%.17g` with no `0.0` special case (so `-0.0` prints `-0`).
//!
//! The decoder is the *strict* parser: it accepts only the named escapes
//! (`\" \\ \/ \b \f \n \r \t`) and rejects `\u` escapes.

use std::sync::{Arc, Mutex};

use lana_bytecode::LanaError;
use lana_vm::value::{Array, Map, MapEntry, Value, ValueKind};

/// Format a finite `f64` exactly as C's `%.17g` (17 significant digits).
///
/// The codec and `data::json_emit` both use `%.17g`; the VM's value printer
/// uses `%.12g` (see `lana_bytecode::format_g`), so this is a separate routine.
pub fn format_17g(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() { "-0".to_string() } else { "0".to_string() };
    }
    let negative = value.is_sign_negative();
    let abs = value.abs();
    // Full-precision scientific form to determine the decimal exponent
    // robustly (avoids the log10 rounding that a direct `log10().floor()`
    // would suffer near powers of ten).
    let sci = format!("{:.*e}", 16, abs);
    let (mantissa, exp) = sci.split_once('e').expect("scientific notation has an exponent");
    let exponent: i32 = exp.parse().expect("exponent is an integer");
    let use_scientific = exponent < -4 || exponent >= 17;
    let mut mantissa = if use_scientific {
        mantissa.to_string()
    } else {
        let decimals = (17 - 1 - exponent).max(0) as usize;
        format!("{:.*}", decimals, abs)
    };
    // Trim trailing zeros and a trailing decimal point, matching `%g`.
    if mantissa.contains('.') {
        while mantissa.ends_with('0') {
            mantissa.pop();
        }
        if mantissa.ends_with('.') {
            mantissa.pop();
        }
    }
    let mut out = if use_scientific {
        let exp_str = format!("{}{:02}", if exponent < 0 { "-" } else { "+" }, exponent.abs());
        format!("{mantissa}e{exp_str}")
    } else {
        mantissa
    };
    if negative {
        out = format!("-{out}");
    }
    out
}

fn append_string(out: &mut String, string: &str) {
    out.push('"');
    for character in string.chars() {
        let replacement = match character {
            '"' => Some("\\\""),
            '\\' => Some("\\\\"),
            '\u{0008}' => Some("\\b"),
            '\u{000c}' => Some("\\f"),
            '\n' => Some("\\n"),
            '\r' => Some("\\r"),
            '\t' => Some("\\t"),
            _ => None,
        };
        if let Some(replacement) = replacement {
            out.push_str(replacement);
        } else if (character as u32) < 0x20 {
            out.push_str(&format!("\\u{:04x}", character as u32));
        } else {
            out.push(character);
        }
    }
    out.push('"');
}

fn encode_value_into(out: &mut String, value: &Value) -> Result<(), LanaError> {
    match &value.kind {
        ValueKind::Null => out.push_str("null"),
        ValueKind::Bool(boolean) => out.push_str(if *boolean { "true" } else { "false" }),
        ValueKind::Number(number) => {
            if !number.is_finite() {
                return Err(LanaError::UnsupportedValue);
            }
            out.push_str(&format_17g(*number));
        }
        ValueKind::String(string) => append_string(out, string),
        ValueKind::Array(array) => {
            out.push('[');
            let array = array.lock().unwrap();
            for (index, item) in array.items.iter().enumerate() {
                if index != 0 {
                    out.push(',');
                }
                encode_value_into(out, item)?;
            }
            out.push(']');
        }
        ValueKind::Map(map) => {
            let map = map.lock().unwrap();
            let mut entries: Vec<&MapEntry> = map.entries.iter().collect();
            entries.sort_by(|a, b| a.key.cmp(&b.key));
            out.push('{');
            for (index, entry) in entries.iter().enumerate() {
                if index != 0 {
                    out.push(',');
                }
                append_string(out, &entry.key);
                out.push(':');
                encode_value_into(out, &entry.value)?;
            }
            out.push('}');
        }
        _ => return Err(LanaError::UnsupportedValue),
    }
    Ok(())
}

/// Encode a value as JSON, mirroring `lana_codec_encode_value`.
pub fn encode_value(value: &Value) -> Result<String, LanaError> {
    let mut out = String::new();
    encode_value_into(&mut out, value)?;
    Ok(out)
}

struct Parser<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Parser<'a> {
    fn skip_space(&mut self) {
        while self.offset < self.data.len() && (self.data[self.offset] as char).is_ascii_whitespace() {
            self.offset += 1;
        }
    }

    fn take(&mut self, expected: u8) -> bool {
        if self.offset < self.data.len() && self.data[self.offset] == expected {
            self.offset += 1;
            true
        } else {
            false
        }
    }
}

fn parse_string(parser: &mut Parser) -> Result<Arc<str>, LanaError> {
    if !parser.take(b'"') {
        return Err(LanaError::Parse);
    }
    let mut out = String::new();
    while parser.offset < parser.data.len() {
        let character = parser.data[parser.offset];
        parser.offset += 1;
        if character == b'"' {
            return Ok(Arc::from(out.as_str()));
        }
        if character < 0x20 {
            return Err(LanaError::Parse);
        }
        if character == b'\\' {
            if parser.offset >= parser.data.len() {
                return Err(LanaError::Parse);
            }
            let escaped = parser.data[parser.offset];
            parser.offset += 1;
            let decoded = match escaped {
                b'"' | b'\\' | b'/' => escaped,
                b'b' => 0x08,
                b'f' => 0x0c,
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                _ => return Err(LanaError::Parse),
            };
            out.push(decoded as char);
        } else {
            out.push(character as char);
        }
    }
    Err(LanaError::Parse)
}

fn parse_value(parser: &mut Parser) -> Result<Value, LanaError> {
    parser.skip_space();
    if parser.offset >= parser.data.len() {
        return Err(LanaError::Parse);
    }
    if parser.data[parser.offset] == b'"' {
        let string = parse_string(parser)?;
        return Ok(Value::string(string));
    }
    let remaining = &parser.data[parser.offset..];
    if remaining.starts_with(b"null") {
        parser.offset += 4;
        return Ok(Value::null());
    }
    if remaining.starts_with(b"true") {
        parser.offset += 4;
        return Ok(Value::boolean(true));
    }
    if remaining.starts_with(b"false") {
        parser.offset += 5;
        return Ok(Value::boolean(false));
    }
    if parser.data[parser.offset] == b'[' {
        parser.offset += 1;
        parser.skip_space();
        let mut items = Vec::new();
        if parser.take(b']') {
            return Ok(Value::array(Arc::new(Mutex::new(Array { items }))));
        }
        loop {
            let item = parse_value(parser)?;
            items.push(item);
            parser.skip_space();
            if parser.take(b']') {
                return Ok(Value::array(Arc::new(Mutex::new(Array { items }))));
            }
            if !parser.take(b',') {
                return Err(LanaError::Parse);
            }
        }
    }
    if parser.data[parser.offset] == b'{' {
        parser.offset += 1;
        parser.skip_space();
        let mut map = Map::new(4);
        if parser.take(b'}') {
            return Ok(Value::map(Arc::new(Mutex::new(map))));
        }
        loop {
            let key = parse_value(parser)?;
            let key = match &key.kind {
                ValueKind::String(string) => string.clone(),
                _ => return Err(LanaError::Schema),
            };
            parser.skip_space();
            if !parser.take(b':') {
                return Err(LanaError::Parse);
            }
            let value = parse_value(parser)?;
            if map.has(&key) {
                return Err(LanaError::Schema);
            }
            map.set(key, value, true).map_err(|_| LanaError::Schema)?;
            parser.skip_space();
            if parser.take(b'}') {
                return Ok(Value::map(Arc::new(Mutex::new(map))));
            }
            if !parser.take(b',') {
                return Err(LanaError::Parse);
            }
        }
    }
    // Number: scan a `strtod`-style decimal prefix (a superset of JSON numbers,
    // matching the C codec's use of `strtod`).
    let (number, end) = parse_number(parser.data, parser.offset)?;
    parser.offset = end;
    Ok(Value::number(number))
}

/// Scan a `strtod`-style decimal number at `offset`, returning the value and
/// the offset just past it. Mirrors the C codec's `strtod` call: an optional
/// sign, digits with an optional fraction, and an optional exponent.
fn parse_number(data: &[u8], offset: usize) -> Result<(f64, usize), LanaError> {
    let n = data.len();
    let mut i = offset;
    if i < n && (data[i] == b'+' || data[i] == b'-') {
        i += 1;
    }
    let mut has_digits = false;
    while i < n && data[i].is_ascii_digit() {
        i += 1;
        has_digits = true;
    }
    if i < n && data[i] == b'.' {
        i += 1;
        while i < n && data[i].is_ascii_digit() {
            i += 1;
            has_digits = true;
        }
    }
    if !has_digits {
        return Err(LanaError::Parse);
    }
    if i < n && (data[i] == b'e' || data[i] == b'E') {
        let mut j = i + 1;
        if j < n && (data[j] == b'+' || data[j] == b'-') {
            j += 1;
        }
        let mut exp_digits = false;
        while j < n && data[j].is_ascii_digit() {
            j += 1;
            exp_digits = true;
        }
        if exp_digits {
            i = j;
        }
    }
    let text = std::str::from_utf8(&data[offset..i]).map_err(|_| LanaError::Parse)?;
    let number: f64 = text.parse().map_err(|_| LanaError::Parse)?;
    if !number.is_finite() {
        return Err(LanaError::Parse);
    }
    Ok((number, i))
}

/// Decode one JSON value, advancing `offset`, mirroring `lana_codec_decode_value`.
pub fn decode_value(data: &[u8], offset: &mut usize) -> Result<Value, LanaError> {
    if *offset > data.len() {
        return Err(LanaError::InvalidState);
    }
    let mut parser = Parser { data, offset: *offset };
    let value = parse_value(&mut parser)?;
    *offset = parser.offset;
    Ok(value)
}

/// Decode a complete JSON document, mirroring `lana_codec_decode_document`.
pub fn decode_document(data: &[u8]) -> Result<Value, LanaError> {
    let mut offset = 0;
    let value = decode_value(data, &mut offset)?;
    while offset < data.len() && (data[offset] as char).is_ascii_whitespace() {
        offset += 1;
    }
    if offset == data.len() {
        Ok(value)
    } else {
        Err(LanaError::Parse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_17g_matches_c() {
        assert_eq!(format_17g(0.0), "0");
        assert_eq!(format_17g(-0.0), "-0");
        assert_eq!(format_17g(0.5), "0.5");
        assert_eq!(format_17g(1.0), "1");
        assert_eq!(format_17g(0.4), "0.40000000000000002");
        assert_eq!(format_17g(1.0 / 3.0), "0.33333333333333331");
        assert_eq!(format_17g(0.1), "0.10000000000000001");
        assert_eq!(format_17g(123.456), "123.456");
        assert_eq!(format_17g(1e-5), "1.0000000000000001e-05");
        assert_eq!(format_17g(1e17), "1e+17");
    }

    #[test]
    fn encode_scalars() {
        assert_eq!(encode_value(&Value::null()).unwrap(), "null");
        assert_eq!(encode_value(&Value::boolean(true)).unwrap(), "true");
        assert_eq!(encode_value(&Value::number(0.4)).unwrap(), "0.40000000000000002");
        assert_eq!(encode_value(&Value::string(Arc::from("hi"))).unwrap(), "\"hi\"");
    }

    #[test]
    fn encode_escapes_control_chars() {
        assert_eq!(
            encode_value(&Value::string(Arc::from("a\nb\tc"))).unwrap(),
            "\"a\\nb\\tc\""
        );
    }

    #[test]
    fn encode_map_sorts_keys() {
        let mut map = Map::new(2);
        map.set(Arc::from("b"), Value::number(2.0), false).unwrap();
        map.set(Arc::from("a"), Value::number(1.0), false).unwrap();
        let value = Value::map(Arc::new(Mutex::new(map)));
        assert_eq!(encode_value(&value).unwrap(), "{\"a\":1,\"b\":2}");
    }

    #[test]
    fn round_trip() {
        let text = br#"{"a":[1,2.5,true,null],"b":"x"}"#;
        let value = decode_document(text).unwrap();
        assert_eq!(encode_value(&value).unwrap(), "{\"a\":[1,2.5,true,null],\"b\":\"x\"}");
    }

    #[test]
    fn decode_rejects_unicode_escape() {
        // codec.c's strict parser does not accept \u escapes.
        assert_eq!(decode_document(br#""\u0041""#).unwrap_err(), LanaError::Parse);
    }
}
