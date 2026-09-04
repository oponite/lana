//! JSON and CSV data boundaries, mirroring `runtime/c/data.c` and `runtime/include/data.h`.
//!
//! This is the *lenient* JSON surface used by the store, ledger, and adapters
//! to decode stored values (`lana_json_parse`) and by host calls to stringify
//! (`lana_json_stringify`). It differs from `codec` in two deliberate ways,
//! both preserved here:
//!
//! * the emitter escapes every control byte as `\u00xx` (no named escapes) and
//!   special-cases `0.0` to `"0"`;
//! * the parser accepts `\u` escapes (including surrogate pairs), enforces a
//!   128-level depth limit, and rejects leading `+`/`0` in numbers.

use std::sync::{Arc, Mutex};

use lana_bytecode::LanaError;
use lana_vm::value::{Array, Map, Value, ValueKind};

use crate::codec::format_17g;

const DATA_DEPTH_LIMIT: usize = 128;

// ---------------------------------------------------------------------------
// JSON parsing (`lana_json_parse`)
// ---------------------------------------------------------------------------

struct JsonParser<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> JsonParser<'a> {
    fn space(&mut self) {
        while self.offset < self.data.len() && (self.data[self.offset] as char).is_ascii_whitespace() {
            self.offset += 1;
        }
    }
}

fn hex_value(c: u8) -> i32 {
    match c {
        b'0'..=b'9' => (c - b'0') as i32,
        b'a'..=b'f' => (c - b'a' + 10) as i32,
        b'A'..=b'F' => (c - b'A' + 10) as i32,
        _ => -1,
    }
}

fn add_utf8(out: &mut String, code: u32) -> bool {
    if code == 0 || code > 0x10ffff || (0xd800..=0xdfff).contains(&code) {
        return false;
    }
    let ch = char::from_u32(code).expect("validated code point");
    out.push(ch);
    true
}

fn json_string(parser: &mut JsonParser) -> Result<Arc<str>, LanaError> {
    if parser.offset >= parser.data.len() || parser.data[parser.offset] != b'"' {
        return Err(LanaError::Parse);
    }
    parser.offset += 1;
    let mut out = String::new();
    while parser.offset < parser.data.len() && parser.data[parser.offset] != b'"' {
        let c = parser.data[parser.offset];
        parser.offset += 1;
        if c < 0x20 {
            return Err(LanaError::Parse);
        }
        if c != b'\\' {
            out.push(c as char);
            continue;
        }
        if parser.offset >= parser.data.len() {
            return Err(LanaError::Parse);
        }
        let escaped = parser.data[parser.offset];
        parser.offset += 1;
        match escaped {
            b'"' | b'\\' | b'/' => out.push(escaped as char),
            b'b' => out.push('\u{0008}'),
            b'f' => out.push('\u{000c}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                if parser.data.len() - parser.offset < 4 {
                    return Err(LanaError::Parse);
                }
                let mut code = 0u32;
                for _ in 0..4 {
                    let h = hex_value(parser.data[parser.offset]);
                    parser.offset += 1;
                    if h < 0 {
                        return Err(LanaError::Parse);
                    }
                    code = (code << 4) | h as u32;
                }
                if (0xd800..=0xdbff).contains(&code) {
                    if parser.data.len() - parser.offset < 6
                        || parser.data[parser.offset] != b'\\'
                        || parser.data[parser.offset + 1] != b'u'
                    {
                        return Err(LanaError::Parse);
                    }
                    parser.offset += 2;
                    let mut low = 0u32;
                    for _ in 0..4 {
                        let h = hex_value(parser.data[parser.offset]);
                        parser.offset += 1;
                        if h < 0 {
                            return Err(LanaError::Parse);
                        }
                        low = (low << 4) | h as u32;
                    }
                    if !(0xdc00..=0xdfff).contains(&low) {
                        return Err(LanaError::Parse);
                    }
                    code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                }
                if !add_utf8(&mut out, code) {
                    return Err(LanaError::Parse);
                }
            }
            _ => return Err(LanaError::Parse),
        }
    }
    if parser.offset >= parser.data.len() {
        return Err(LanaError::Parse);
    }
    parser.offset += 1;
    Ok(Arc::from(out.as_str()))
}

fn json_value(parser: &mut JsonParser, depth: usize) -> Result<Value, LanaError> {
    if depth > DATA_DEPTH_LIMIT {
        return Err(LanaError::Limit);
    }
    parser.space();
    if parser.offset >= parser.data.len() {
        return Err(LanaError::Parse);
    }
    if parser.data[parser.offset] == b'"' {
        let string = json_string(parser)?;
        return Ok(Value::string(string));
    }
    if parser.data[parser.offset] == b'[' {
        parser.offset += 1;
        parser.space();
        let mut items = Vec::new();
        while parser.offset < parser.data.len() && parser.data[parser.offset] != b']' {
            let item = json_value(parser, depth + 1)?;
            items.push(item);
            parser.space();
            if parser.offset < parser.data.len() && parser.data[parser.offset] == b',' {
                parser.offset += 1;
                parser.space();
                if parser.offset < parser.data.len() && parser.data[parser.offset] == b']' {
                    return Err(LanaError::Parse);
                }
            } else {
                break;
            }
        }
        if parser.offset >= parser.data.len() || parser.data[parser.offset] != b']' {
            return Err(LanaError::Parse);
        }
        parser.offset += 1;
        return Ok(Value::array(Arc::new(Mutex::new(Array { items }))));
    }
    if parser.data[parser.offset] == b'{' {
        parser.offset += 1;
        parser.space();
        let mut map = Map::new(4);
        while parser.offset < parser.data.len() && parser.data[parser.offset] != b'}' {
            let key = json_string(parser)?;
            parser.space();
            if parser.offset >= parser.data.len() || parser.data[parser.offset] != b':' {
                return Err(LanaError::Parse);
            }
            parser.offset += 1;
            let item = json_value(parser, depth + 1)?;
            if map.set(key, item, true).is_err() {
                return Err(LanaError::Parse);
            }
            parser.space();
            if parser.offset < parser.data.len() && parser.data[parser.offset] == b',' {
                parser.offset += 1;
                parser.space();
                if parser.offset < parser.data.len() && parser.data[parser.offset] == b'}' {
                    return Err(LanaError::Parse);
                }
            } else {
                break;
            }
        }
        if parser.offset >= parser.data.len() || parser.data[parser.offset] != b'}' {
            return Err(LanaError::Parse);
        }
        parser.offset += 1;
        return Ok(Value::map(Arc::new(Mutex::new(map))));
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
    // Number: `strtod` plus the strict leading `+`/`0` rejection.
    let (number, end) = parse_number_strict(parser.data, parser.offset)?;
    parser.offset = end;
    Ok(Value::number(number))
}

fn parse_number_strict(data: &[u8], offset: usize) -> Result<(f64, usize), LanaError> {
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
    // Reject a leading `+`, a leading `0` followed by a digit, or `-0` followed
    // by a digit, matching `lana_json_parse`.
    if data[offset] == b'+'
        || (data[offset] == b'0' && i > offset + 1 && data[offset + 1].is_ascii_digit())
        || (data[offset] == b'-'
            && offset + 2 < i
            && data[offset + 1] == b'0'
            && data[offset + 2].is_ascii_digit())
    {
        return Err(LanaError::Parse);
    }
    let text = std::str::from_utf8(&data[offset..i]).map_err(|_| LanaError::Parse)?;
    let number: f64 = text.parse().map_err(|_| LanaError::Parse)?;
    if !number.is_finite() {
        return Err(LanaError::Parse);
    }
    Ok((number, i))
}

/// Parse a complete JSON document, mirroring `lana_json_parse`.
pub fn json_parse(text: &str) -> Result<Value, LanaError> {
    let data = text.as_bytes();
    let mut parser = JsonParser { data, offset: 0 };
    let value = json_value(&mut parser, 0)?;
    parser.space();
    if parser.offset != data.len() {
        return Err(LanaError::Parse);
    }
    Ok(value)
}

// ---------------------------------------------------------------------------
// JSON stringify (`lana_json_stringify`)
// ---------------------------------------------------------------------------

fn json_escape(out: &mut String, text: &str) -> bool {
    out.push('"');
    for c in text.chars() {
        if c == '"' || c == '\\' {
            out.push('\\');
            out.push(c);
        } else if (c as u32) < 0x20 {
            out.push_str(&format!("\\u{:04x}", c as u32));
        } else {
            out.push(c);
        }
    }
    out.push('"');
    true
}

fn json_emit(value: &Value, out: &mut String, stack: &mut Vec<usize>, depth: usize) -> Result<(), LanaError> {
    if depth > DATA_DEPTH_LIMIT {
        return Err(LanaError::Limit);
    }
    let identity = match &value.kind {
        ValueKind::Array(array) => Some(Arc::as_ptr(array) as usize),
        ValueKind::Map(map) => Some(Arc::as_ptr(map) as usize),
        _ => None,
    };
    if let Some(identity) = identity {
        if stack[..depth].contains(&identity) {
            return Err(LanaError::UnsupportedOperation);
        }
        stack[depth] = identity;
    }
    match &value.kind {
        ValueKind::Null => out.push_str("null"),
        ValueKind::Bool(boolean) => out.push_str(if *boolean { "true" } else { "false" }),
        ValueKind::Number(number) => {
            if !number.is_finite() {
                return Err(LanaError::UnsupportedOperation);
            }
            if *number == 0.0 {
                out.push('0');
            } else {
                out.push_str(&format_17g(*number));
            }
        }
        ValueKind::String(string) => {
            json_escape(out, string);
        }
        ValueKind::Array(array) => {
            out.push('[');
            let array = array.lock().unwrap();
            for (index, item) in array.items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                json_emit(item, out, stack, depth + 1)?;
            }
            out.push(']');
        }
        ValueKind::Map(map) => {
            out.push('{');
            let map = map.lock().unwrap();
            for (index, entry) in map.entries.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                json_escape(out, &entry.key);
                out.push(':');
                json_emit(&entry.value, out, stack, depth + 1)?;
            }
            out.push('}');
        }
        _ => return Err(LanaError::UnsupportedOperation),
    }
    Ok(())
}

/// Stringify a value as JSON, mirroring `lana_json_stringify`.
pub fn json_stringify(value: &Value) -> Result<String, LanaError> {
    let mut out = String::new();
    let mut stack = vec![0usize; DATA_DEPTH_LIMIT + 1];
    json_emit(value, &mut out, &mut stack, 0)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// CSV (`lana_csv_read` / `lana_csv_write`)
// ---------------------------------------------------------------------------

fn csv_records(text: &str) -> Result<Vec<Vec<String>>, LanaError> {
    let bytes = text.as_bytes();
    let length = bytes.len();
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut fields: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut i = 0usize;
    while i < length {
        let mut quoted = false;
        if bytes[i] == b'"' {
            quoted = true;
            i += 1;
            while i < length {
                if bytes[i] == b'"' {
                    if i + 1 < length && bytes[i + 1] == b'"' {
                        field.push('"');
                        i += 2;
                    } else {
                        i += 1;
                        break;
                    }
                } else {
                    field.push(bytes[i] as char);
                    i += 1;
                }
            }
            if i > length || (i == length && (length == 0 || bytes[length - 1] != b'"')) {
                return Err(LanaError::Parse);
            }
        } else {
            while i < length && bytes[i] != b',' && bytes[i] != b'\r' && bytes[i] != b'\n' {
                if bytes[i] == b'"' {
                    return Err(LanaError::Parse);
                }
                field.push(bytes[i] as char);
                i += 1;
            }
        }
        if quoted && i < length && bytes[i] != b',' && bytes[i] != b'\r' && bytes[i] != b'\n' {
            return Err(LanaError::Parse);
        }
        fields.push(std::mem::take(&mut field));
        if i < length && bytes[i] == b',' {
            i += 1;
            if i == length {
                fields.push(String::new());
            }
            continue;
        }
        if i < length && bytes[i] == b'\r' {
            if i + 1 >= length || bytes[i + 1] != b'\n' {
                return Err(LanaError::Parse);
            }
            i += 2;
        } else if i < length && bytes[i] == b'\n' {
            i += 1;
        }
        records.push(std::mem::take(&mut fields));
    }
    Ok(records)
}

/// Read a CSV file into an array of maps, mirroring `lana_csv_read`.
pub fn csv_read(path: &str) -> Result<Value, LanaError> {
    let mut text = std::fs::read_to_string(path).map_err(|_| LanaError::Io)?;
    if text.starts_with('\u{feff}') {
        text.drain(..3);
    }
    let records = csv_records(&text)?;
    if let Some(header) = records.first() {
        for (column, name) in header.iter().enumerate() {
            if name.is_empty() {
                return Err(LanaError::Parse);
            }
            for other in &header[..column] {
                if other == name {
                    return Err(LanaError::Parse);
                }
            }
        }
    }
    let row_count = if records.is_empty() { 0 } else { records.len() - 1 };
    let mut items = Vec::with_capacity(row_count);
    if let Some(header) = records.first() {
        for row in &records[1..] {
            if row.len() != header.len() {
                return Err(LanaError::Parse);
            }
            let mut map = Map::new(header.len());
            for (column, name) in header.iter().enumerate() {
                let value = Value::string(Arc::from(row[column].as_str()));
                map.set(Arc::from(name.as_str()), value, true).map_err(|_| LanaError::Parse)?;
            }
            items.push(Value::map(Arc::new(Mutex::new(map))));
        }
    }
    Ok(Value::array(Arc::new(Mutex::new(Array { items }))))
}

fn csv_scalar(value: &Value, field: &mut String) -> bool {
    let text = match &value.kind {
        ValueKind::Null => String::new(),
        ValueKind::String(string) => string.to_string(),
        ValueKind::Bool(boolean) => if *boolean { "true".to_string() } else { "false".to_string() },
        ValueKind::Number(number) if number.is_finite() => {
            if *number == 0.0 {
                "0".to_string()
            } else {
                format_17g(*number)
            }
        }
        _ => return false,
    };
    if !text.contains([',', '"', '\r', '\n']) {
        field.push_str(&text);
        return true;
    }
    field.push('"');
    for c in text.chars() {
        if c == '"' {
            field.push('"');
        }
        field.push(c);
    }
    field.push('"');
    true
}

/// Write an array of maps as CSV, mirroring `lana_csv_write`.
pub fn csv_write(path: &str, rows: &Value) -> Result<Value, LanaError> {
    let array = match &rows.kind {
        ValueKind::Array(array) => array,
        _ => return Err(LanaError::Type),
    };
    let array = array.lock().unwrap();
    if array.items.is_empty() {
        return Err(LanaError::Type);
    }
    // Extract the header keys up front (releasing the map lock) so the row
    // loop below can re-lock the same map without deadlocking.
    let header_keys: Vec<Arc<str>> = match &array.items[0].kind {
        ValueKind::Map(map) => map.lock().unwrap().entries.iter().map(|e| e.key.clone()).collect(),
        _ => return Err(LanaError::Type),
    };
    let mut out = String::new();
    for row in 0..=array.items.len() {
        let row_map = if row == 0 {
            None
        } else {
            match &array.items[row - 1].kind {
                ValueKind::Map(map) => Some(map.lock().unwrap()),
                _ => return Err(LanaError::Type),
            }
        };
        if let Some(map) = row_map.as_ref() {
            if map.entries.len() != header_keys.len() {
                return Err(LanaError::Type);
            }
        }
        for (column, key) in header_keys.iter().enumerate() {
            if column > 0 {
                out.push(',');
            }
            let mut field = String::new();
            if row == 0 {
                let key_value = Value::string(key.clone());
                if !csv_scalar(&key_value, &mut field) {
                    return Err(LanaError::Type);
                }
            } else {
                let map = row_map.as_ref().unwrap();
                if map.entries[column].key != *key {
                    return Err(LanaError::Type);
                }
                if !csv_scalar(&map.entries[column].value, &mut field) {
                    return Err(LanaError::Type);
                }
            }
            out.push_str(&field);
        }
        out.push_str("\r\n");
    }
    std::fs::write(path, out.as_bytes()).map_err(|_| LanaError::Io)?;
    Ok(Value::boolean(true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_parse_handles_unicode_escape() {
        let value = json_parse(r#""A""#).unwrap();
        assert_eq!(value.as_string(), Arc::from("A"));
    }

    #[test]
    fn json_parse_rejects_leading_zero() {
        assert_eq!(json_parse("01").unwrap_err(), LanaError::Parse);
        assert_eq!(json_parse("+1").unwrap_err(), LanaError::Parse);
    }

    #[test]
    fn json_stringify_escapes_control_as_unicode() {
        let value = Value::string(Arc::from("a\nb"));
        assert_eq!(json_stringify(&value).unwrap(), "\"a\\u000ab\"");
    }

    #[test]
    fn json_stringify_zero() {
        assert_eq!(json_stringify(&Value::number(0.0)).unwrap(), "0");
        assert_eq!(json_stringify(&Value::number(-0.0)).unwrap(), "0");
    }

    #[test]
    fn csv_round_trip() {
        let mut map = Map::new(2);
        map.set(Arc::from("a"), Value::string(Arc::from("1")), false).unwrap();
        map.set(Arc::from("b"), Value::string(Arc::from("x,y")), false).unwrap();
        let rows = Value::array(Arc::new(Mutex::new(Array {
            items: vec![Value::map(Arc::new(Mutex::new(map)))],
        })));
        let path = std::env::temp_dir().join("lana_csv_test.csv");
        let path = path.to_str().unwrap();
        csv_write(path, &rows).unwrap();
        let read = csv_read(path).unwrap();
        let _ = std::fs::remove_file(path);
        match &read.kind {
            ValueKind::Array(array) => {
                let array = array.lock().unwrap();
                assert_eq!(array.items.len(), 1);
            }
            _ => panic!("expected array"),
        }
    }
}
