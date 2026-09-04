//! Hardware adapters, mirroring `runtime/c/adapters.c` and `runtime/include/adapters.h`.
//!
//! JSON and CSV are in-core adapters; SQLite and HTTP_JSON are dlopen plugins
//! in the C implementation. The Rust port has no plugin mechanism, so those two
//! kinds fail to load with `LanaError::Io` (the same code the C path returns
//! when a plugin library is missing) and can never be fetched.

use std::sync::Arc;

use lana_bytecode::LanaError;
use lana_vm::value::Value;

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

/// A loaded adapter. JSON and CSV are in-core and carry no plugin handle.
#[derive(Debug)]
pub struct Adapter {
    kind: AdapterKind,
}

/// Load an adapter, mirroring `lana_adapter_load`.
///
/// JSON and CSV are in-core; SQLite and HTTP_JSON are plugins in the C
/// implementation and return `LanaError::Io` here (the "plugin library missing"
/// path). An invalid schema version returns `LanaError::InvalidState`.
pub fn adapter_load(options: &AdapterOptions) -> Result<Adapter, LanaError> {
    if options.schema_version != 1 {
        return Err(LanaError::InvalidState);
    }
    match options.kind {
        AdapterKind::Json | AdapterKind::Csv => Ok(Adapter { kind: options.kind }),
        AdapterKind::Sqlite | AdapterKind::HttpJson => Err(LanaError::Io),
    }
}

/// Fetch evidence through an adapter, mirroring `lana_adapter_fetch`.
///
/// For JSON the query is a JSON document; for CSV it is a file path. SQLite and
/// HTTP_JSON can never be loaded in the Rust port, so they return
/// `LanaError::UnsupportedOperation`.
pub fn adapter_fetch(adapter: &Adapter, query: &str) -> Result<Value, LanaError> {
    match adapter.kind {
        AdapterKind::Json => data::json_parse(query),
        AdapterKind::Csv => data::csv_read(query),
        AdapterKind::Sqlite | AdapterKind::HttpJson => Err(LanaError::UnsupportedOperation),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lana_vm::value::ValueKind;

    #[test]
    fn load_json_succeeds() {
        let options = AdapterOptions {
            schema_version: 1,
            kind: AdapterKind::Json,
            config: None,
        };
        let adapter = adapter_load(&options).unwrap();
        assert_eq!(adapter.kind, AdapterKind::Json);
    }

    #[test]
    fn load_sqlite_returns_io() {
        let options = AdapterOptions {
            schema_version: 1,
            kind: AdapterKind::Sqlite,
            config: None,
        };
        assert_eq!(adapter_load(&options).unwrap_err(), LanaError::Io);
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
    fn fetch_json_parses_map() {
        let adapter = Adapter { kind: AdapterKind::Json };
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
        let adapter = Adapter { kind: AdapterKind::Csv };
        let value = adapter_fetch(&adapter, path_str).unwrap();
        let _ = std::fs::remove_file(path_str);
        match &value.kind {
            ValueKind::Array(array) => {
                let array = array.lock().unwrap();
                assert_eq!(array.items.len(), 1);
            }
            _ => panic!("expected array"),
        }
    }
}
