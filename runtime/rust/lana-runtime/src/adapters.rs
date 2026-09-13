//! Hardware adapters, mirroring `runtime/c/adapters.c`, the dlopen plugins in
//! `runtime/c/adapters/`, and `runtime/include/adapters.h`.
//!
//! JSON and CSV are in-core adapters. SQLite and HTTP_JSON are dlopen plugins
//! in the C implementation; the Rust port has no plugin mechanism, so it
//! implements both natively here with the same error codes and value shapes.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use lana_vm::gc::{Gc, GraphCell};

use lana_bytecode::LanaError;
use lana_vm::value::{Value, ValueKind};
#[cfg(not(target_arch = "wasm32"))]
use lana_vm::value::Map;
#[cfg(not(target_arch = "wasm32"))]
use rusqlite::types::ValueRef;
#[cfg(not(target_arch = "wasm32"))]
use rusqlite::{Connection, OpenFlags};

use crate::data;

/// The adapter kind, matching `LanaAdapterKind` in `runtime/include/adapters.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AdapterKind {
    Json = 0,
    Csv,
    Sqlite,
    HttpJson,
}

/// Adapter load options, matching `LanaAdapterOptions` in
/// `runtime/include/adapters.h`. The C struct carries a `struct_size` field for
/// ABI versioning; the Rust port uses the type system instead.
pub struct AdapterOptions {
    pub schema_version: u32,
    pub kind: AdapterKind,
    pub config: Option<Arc<str>>,
}

/// A loaded adapter. JSON and CSV carry no state; SQLite holds the open
/// read-only connection and HTTP_JSON holds the evidence-server port.
#[derive(Debug)]
pub enum Adapter {
    Json,
    Csv,
    #[cfg(not(target_arch = "wasm32"))]
    Sqlite(Connection),
    HttpJson(u16),
}

/// Load an adapter, mirroring `lana_adapter_load`.
///
/// JSON and CSV are in-core. SQLite opens the database at `config` read-only
/// (`SQLITE_OPEN_READONLY`); a missing `config` returns `LanaError::Schema`
/// (the C plugin's null-config code) and an open failure returns
/// `LanaError::Io`. HTTP_JSON stores the port parsed from `config`, defaulting
/// to 8080 when absent. An invalid schema version returns
/// `LanaError::InvalidState`.
pub fn adapter_load(options: &AdapterOptions) -> Result<Adapter, LanaError> {
    if options.schema_version != 1 {
        return Err(LanaError::InvalidState);
    }
    match options.kind {
        AdapterKind::Json => Ok(Adapter::Json),
        AdapterKind::Csv => Ok(Adapter::Csv),
        #[cfg(target_arch = "wasm32")]
        AdapterKind::Sqlite => Err(LanaError::UnsupportedOperation),
        #[cfg(not(target_arch = "wasm32"))]
        AdapterKind::Sqlite => {
            let path = options.config.as_deref().ok_or(LanaError::Schema)?;
            let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .map_err(|_| LanaError::Io)?;
            Ok(Adapter::Sqlite(conn))
        }
        AdapterKind::HttpJson => {
            let port = options.config.as_deref().map_or(8080u16, atoi_port);
            Ok(Adapter::HttpJson(port))
        }
    }
}

/// Fetch evidence through an adapter, mirroring `lana_adapter_fetch`.
///
/// For JSON the query is a JSON document; for CSV it is a file path. SQLite
/// prepares `query` and maps the first result row to a map keyed by column
/// name (no row maps to `null`). HTTP_JSON performs a GET to
/// `127.0.0.1:<port>/<query>` and decodes the evidence envelope.
pub fn adapter_fetch(adapter: &Adapter, query: &str) -> Result<Value, LanaError> {
    match adapter {
        Adapter::Json => data::json_parse(query),
        Adapter::Csv => data::csv_read(query),
        #[cfg(not(target_arch = "wasm32"))]
        Adapter::Sqlite(conn) => fetch_sqlite(conn, query),
        Adapter::HttpJson(port) => fetch_http_json(*port, query),
    }
}

// ---------------------------------------------------------------------------
// SQLite (`runtime/c/adapters/sqlite_adapter.c`)
// ---------------------------------------------------------------------------

/// Run `query` against an already-open read-only connection and map the first
/// result row, mirroring `lana_adapter_plugin_fetch` in `sqlite_adapter.c`.
#[cfg(not(target_arch = "wasm32"))]
fn fetch_sqlite(conn: &Connection, query: &str) -> Result<Value, LanaError> {
    let mut stmt = conn.prepare(query).map_err(|_| LanaError::Parse)?;
    // C calls `sqlite3_step` once and treats anything but SQLITE_ROW (a missing
    // row, a step error, an unbound parameter) as "no evidence": null.
    let mut rows = match stmt.query([]) {
        Ok(rows) => rows,
        Err(_) => return Ok(Value::null()),
    };
    let row = match rows.next() {
        Ok(Some(row)) => row,
        Ok(None) | Err(_) => return Ok(Value::null()),
    };
    let column_count = row.as_ref().column_count();
    let heap = lana_vm::heap::Heap::new(256 * 1024 * 1024);
    let mut map = Map::new(&heap, column_count)?;
    for index in 0..column_count {
        // `sqlite3_column_name` returns NULL only on OOM/out-of-range; the C
        // plugin hands that NULL key to `lana_map_set`, which rejects it with
        // LANA_ERR_TYPE.
        let name = match row.as_ref().column_name(index) {
            Ok(name) => Arc::from(name),
            Err(_) => return Err(LanaError::Type),
        };
        let value = match row.get_ref(index) {
            Ok(ValueRef::Null) => Value::null(),
            Ok(ValueRef::Integer(integer)) => Value::number(integer as f64),
            Ok(ValueRef::Real(real)) => Value::number(real),
            // `sqlite3_column_text` + `strlen`: take the bytes up to the first
            // NUL. TEXT/BLOB share this path in the C plugin.
            Ok(ValueRef::Text(bytes)) | Ok(ValueRef::Blob(bytes)) => {
                let length = bytes.iter().position(|&byte| byte == 0).unwrap_or(bytes.len());
                Value::string(Arc::from(String::from_utf8_lossy(&bytes[..length]).as_ref()))
            }
            Err(_) => Value::null(),
        };
        map.set(name, value, true)?;
    }
    Ok(Value::map(Gc::new(&heap, GraphCell::new(map))?))
}

// ---------------------------------------------------------------------------
// HTTP_JSON (`runtime/c/adapters/http_json_adapter.c`)
// ---------------------------------------------------------------------------

/// The HTTP response buffer cap, mirroring `char response[65536]` in
/// `http_json_adapter.c` (reads at most `sizeof(response) - 1` bytes).
const HTTP_RESPONSE_CAP: usize = 65535;

/// Parse the adapter port with `atoi` semantics (leading whitespace, optional
/// sign, decimal digits, trailing garbage ignored, no digits -> 0), then
/// truncate to `uint16_t` exactly as `htons((uint16_t)port)` does in C.
fn atoi_port(config: &str) -> u16 {
    let bytes = config.as_bytes();
    let mut index = 0;
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    let mut negative = false;
    if index < bytes.len() && (bytes[index] == b'+' || bytes[index] == b'-') {
        negative = bytes[index] == b'-';
        index += 1;
    }
    let mut value: i64 = 0;
    let mut any = false;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        any = true;
        value = value
            .saturating_mul(10)
            .saturating_add((bytes[index] - b'0') as i64);
        index += 1;
    }
    if !any {
        return 0;
    }
    if negative {
        value = value.wrapping_neg();
    }
    value as u16
}

/// Build the exact request line the C adapter writes, for testing.
fn http_get_request(query: &str) -> String {
    format!("GET /{query} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
}

/// Map an evidence-server error code to `LanaError`, mirroring `map_error_code`.
fn map_error_code(code: &str) -> LanaError {
    match code {
        "LANA_ERR_NOT_FOUND" => LanaError::NotFound,
        "LANA_ERR_LIMIT" => LanaError::Limit,
        "LANA_ERR_PARSE" => LanaError::Parse,
        "LANA_ERR_UNSUPPORTED_OPERATION" => LanaError::UnsupportedOperation,
        _ => LanaError::Io,
    }
}

/// Perform a plain HTTP GET to the loopback evidence server and decode the
/// `{"schema":1,"ok":...,"evidence":...}` envelope, mirroring
/// `lana_adapter_plugin_fetch` in `http_json_adapter.c`.
fn fetch_http_json(port: u16, query: &str) -> Result<Value, LanaError> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).map_err(|_| LanaError::Io)?;
    stream
        .write_all(http_get_request(query).as_bytes())
        .map_err(|_| LanaError::Io)?;
    let mut response = Vec::new();
    let mut buffer = [0u8; 8192];
    loop {
        if response.len() >= HTTP_RESPONSE_CAP {
            break;
        }
        let want = (HTTP_RESPONSE_CAP - response.len()).min(buffer.len());
        match stream.read(&mut buffer[..want]) {
            Ok(0) | Err(_) => break, // EOF or read error: stop, like C's `n <= 0`.
            Ok(read) => response.extend_from_slice(&buffer[..read]),
        }
    }
    if response.is_empty() {
        return Err(LanaError::Io);
    }
    let body = find_header_end(&response).ok_or(LanaError::Parse)?;
    let body = std::str::from_utf8(body).map_err(|_| LanaError::Parse)?;
    let parsed = data::json_parse(body)?;
    let map = match &parsed.kind {
        ValueKind::Map(map) => map.clone(),
        _ => return Err(LanaError::Corruption),
    };
    let map = map.lock().unwrap();
    let ok = match map.get("ok") {
        Some(Value { kind: ValueKind::Bool(ok), .. }) => *ok,
        _ => return Err(LanaError::Corruption),
    };
    if !ok {
        let error = match map.get("error") {
            Some(Value { kind: ValueKind::Map(error), .. }) => error.clone(),
            _ => return Err(LanaError::Corruption),
        };
        let error = error.lock().unwrap();
        let code = match error.get("code") {
            Some(Value { kind: ValueKind::String(code), .. }) => code.clone(),
            _ => return Err(LanaError::Corruption),
        };
        return Err(map_error_code(&code));
    }
    match map.get("evidence") {
        Some(evidence) => Ok(evidence.clone()),
        None => Err(LanaError::Corruption),
    }
}

/// Return the body after the first `\r\n\r\n`, mirroring `strstr` + `body += 4`.
fn find_header_end(response: &[u8]) -> Option<&[u8]> {
    const MARKER: &[u8] = b"\r\n\r\n";
    response
        .windows(MARKER.len())
        .position(|window| window == MARKER)
        .map(|offset| &response[offset + MARKER.len()..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    fn temp_db_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    #[test]
    fn load_json_succeeds() {
        let options = AdapterOptions {
            schema_version: 1,
            kind: AdapterKind::Json,
            config: None,
        };
        match adapter_load(&options).unwrap() {
            Adapter::Json => {}
            _ => panic!("expected json adapter"),
        }
    }

    #[test]
    fn load_invalid_schema_version_returns_invalid_state() {
        let options = AdapterOptions {
            schema_version: 0,
            kind: AdapterKind::Json,
            config: None,
        };
        assert_eq!(adapter_load(&options).unwrap_err(), LanaError::InvalidState);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn load_sqlite_without_config_returns_schema() {
        let options = AdapterOptions {
            schema_version: 1,
            kind: AdapterKind::Sqlite,
            config: None,
        };
        assert_eq!(adapter_load(&options).unwrap_err(), LanaError::Schema);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn load_sqlite_missing_file_returns_io() {
        let options = AdapterOptions {
            schema_version: 1,
            kind: AdapterKind::Sqlite,
            config: Some(Arc::from("/nonexistent/lana/missing.db")),
        };
        assert_eq!(adapter_load(&options).unwrap_err(), LanaError::Io);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn fetch_sqlite_reads_first_row() {
        let path = temp_db_path("lana_adapter_sqlite_test.db");
        let path_str = path.to_str().unwrap();
        let _ = std::fs::remove_file(path_str);
        {
            let conn = Connection::open(path_str).unwrap();
            conn.execute_batch(
                "CREATE TABLE t (id INTEGER, name TEXT, score REAL, data BLOB); \
                 INSERT INTO t VALUES (1, 'alice', 3.5, X'6162');",
            )
            .unwrap();
        }
        let options = AdapterOptions {
            schema_version: 1,
            kind: AdapterKind::Sqlite,
            config: Some(Arc::from(path_str)),
        };
        let adapter = adapter_load(&options).unwrap();
        let value = adapter_fetch(&adapter, "SELECT * FROM t").unwrap();
        let _ = std::fs::remove_file(path_str);
        let ValueKind::Map(map) = &value.kind else { panic!("expected map") };
        let map = map.lock().unwrap();
        assert_eq!(map.get("id").map(|v| v.as_number()), Some(1.0));
        assert_eq!(map.get("name").map(|v| v.as_string()), Some(Arc::from("alice")));
        assert_eq!(map.get("score").map(|v| v.as_number()), Some(3.5));
        assert_eq!(map.get("data").map(|v| v.as_string()), Some(Arc::from("ab")));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn fetch_sqlite_null_empty_and_bad_sql() {
        let path = temp_db_path("lana_adapter_sqlite_null_test.db");
        let path_str = path.to_str().unwrap();
        let _ = std::fs::remove_file(path_str);
        {
            let conn = Connection::open(path_str).unwrap();
            conn.execute_batch("CREATE TABLE t (id INTEGER, note TEXT); INSERT INTO t VALUES (NULL, NULL);")
                .unwrap();
        }
        let options = AdapterOptions {
            schema_version: 1,
            kind: AdapterKind::Sqlite,
            config: Some(Arc::from(path_str)),
        };
        let adapter = adapter_load(&options).unwrap();

        let value = adapter_fetch(&adapter, "SELECT id, note FROM t").unwrap();
        let ValueKind::Map(map) = &value.kind else { panic!("expected map") };
        let map = map.lock().unwrap();
        assert!(matches!(map.get("id").map(|v| &v.kind), Some(ValueKind::Null)));
        assert!(matches!(map.get("note").map(|v| &v.kind), Some(ValueKind::Null)));

        // No matching row -> null.
        let empty = adapter_fetch(&adapter, "SELECT id FROM t WHERE id > 100").unwrap();
        assert!(matches!(empty.kind, ValueKind::Null));

        // Prepare failure -> Parse.
        assert_eq!(adapter_fetch(&adapter, "NOT VALID SQL").unwrap_err(), LanaError::Parse);

        let _ = std::fs::remove_file(path_str);
    }

    #[test]
    fn load_http_json_port_parsing() {
        let options = AdapterOptions {
            schema_version: 1,
            kind: AdapterKind::HttpJson,
            config: None,
        };
        match adapter_load(&options).unwrap() {
            Adapter::HttpJson(port) => assert_eq!(port, 8080),
            _ => panic!("expected http_json adapter"),
        }
        assert_eq!(atoi_port("9090"), 9090);
        assert_eq!(atoi_port("123abc"), 123);
        assert_eq!(atoi_port(""), 0);
        assert_eq!(atoi_port("no digits"), 0);
        assert_eq!(atoi_port("-1"), 65535);
    }

    #[test]
    fn http_error_mapping_and_request_format() {
        assert_eq!(map_error_code("LANA_ERR_NOT_FOUND"), LanaError::NotFound);
        assert_eq!(map_error_code("LANA_ERR_LIMIT"), LanaError::Limit);
        assert_eq!(map_error_code("LANA_ERR_PARSE"), LanaError::Parse);
        assert_eq!(
            map_error_code("LANA_ERR_UNSUPPORTED_OPERATION"),
            LanaError::UnsupportedOperation
        );
        assert_eq!(map_error_code("LANA_ERR_IO"), LanaError::Io);
        assert_eq!(map_error_code("anything else"), LanaError::Io);

        assert_eq!(
            http_get_request("evidence/abc"),
            "GET /evidence/abc HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        );
    }

    #[test]
    fn fetch_json_parses_map() {
        let adapter = Adapter::Json;
        let value = adapter_fetch(&adapter, r#"{"a":1}"#).unwrap();
        match &value.kind {
            ValueKind::Map(map) => {
                let map = map.lock().unwrap();
                assert_eq!(map.get("a").map(|v| v.as_number()), Some(1.0));
            }
            _ => panic!("expected map"),
        }
    }

    #[test]
    fn fetch_csv_reads_file() {
        let path = std::env::temp_dir().join("lana_adapter_csv_test.csv");
        let path_str = path.to_str().unwrap();
        std::fs::write(path_str, "a,b\r\n1,2\r\n").unwrap();
        let adapter = Adapter::Csv;
        let value = adapter_fetch(&adapter, path_str).unwrap();
        let _ = std::fs::remove_file(path_str);
        match &value.kind {
            ValueKind::Array(array) => {
                let array = array.lock().unwrap();
                assert_eq!(array.items().len(), 1);
            }
            _ => panic!("expected array"),
        }
    }
}
