//! Explicit JSON, CSV, SQLite, and localhost HTTP evidence adapters.

use std::sync::Arc;

use lana_bytecode::LanaError;
use lana_vm::value::{Value, ValueKind};

use crate::data;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AdapterKind {
    Json = 0,
    Csv,
    Sqlite,
    HttpJson,
}

pub struct AdapterOptions {
    pub schema_version: u32,
    pub kind: AdapterKind,
    pub config: Option<Arc<str>>,
}

#[derive(Debug)]
pub struct Adapter {
    kind: AdapterKind,
    #[cfg(not(target_arch = "wasm32"))]
    sqlite: Option<rusqlite::Connection>,
    port: Option<u16>,
}

pub fn adapter_load(options: &AdapterOptions) -> Result<Adapter, LanaError> {
    if options.schema_version != 1 {
        return Err(LanaError::InvalidState);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let mut sqlite = None;
    let mut port = None;
    match options.kind {
        AdapterKind::Json | AdapterKind::Csv => {}
        AdapterKind::Sqlite => {
            #[cfg(target_arch = "wasm32")]
            return Err(LanaError::UnsupportedOperation);
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rusqlite::OpenFlags;
                let path = options.config.as_deref().filter(|path| !path.is_empty()).ok_or(LanaError::Schema)?;
                sqlite = Some(rusqlite::Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|_| LanaError::Io)?);
            }
        }
        AdapterKind::HttpJson => {
            #[cfg(target_arch = "wasm32")]
            return Err(LanaError::UnsupportedOperation);
            #[cfg(not(target_arch = "wasm32"))]
            {
                port = Some(match options.config.as_deref().filter(|text| !text.is_empty()) {
                    Some(text) => text.parse::<u16>().ok().filter(|port| *port != 0).ok_or(LanaError::Schema)?,
                    None => 8080,
                });
            }
        }
    }
    Ok(Adapter {
        kind: options.kind,
        #[cfg(not(target_arch = "wasm32"))]
        sqlite,
        port,
    })
}

pub fn adapter_fetch(adapter: &Adapter, query: &str) -> Result<Value, LanaError> {
    match adapter.kind {
        AdapterKind::Json => data::json_parse(query),
        AdapterKind::Csv => data::csv_read(query),
        AdapterKind::Sqlite => {
            #[cfg(target_arch = "wasm32")]
            return Err(LanaError::UnsupportedOperation);
            #[cfg(not(target_arch = "wasm32"))]
            return sqlite_fetch(adapter.sqlite.as_ref().ok_or(LanaError::InvalidState)?, query);
        }
        AdapterKind::HttpJson => {
            #[cfg(target_arch = "wasm32")]
            return Err(LanaError::UnsupportedOperation);
            #[cfg(not(target_arch = "wasm32"))]
            return http_fetch(adapter.port.ok_or(LanaError::InvalidState)?, query);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn sqlite_fetch(connection: &rusqlite::Connection, query: &str) -> Result<Value, LanaError> {
    use rusqlite::types::ValueRef;
    let mut statement = connection.prepare(query).map_err(|_| LanaError::Parse)?;
    if !statement.readonly() {
        return Err(LanaError::UnsupportedOperation);
    }
    let names: Vec<String> = (0..statement.column_count())
        .map(|index| statement.column_name(index).map(str::to_string).map_err(|_| LanaError::Parse))
        .collect::<Result<_, _>>()?;
    let mut rows = statement.query([]).map_err(|_| LanaError::Io)?;
    let Some(row) = rows.next().map_err(|_| LanaError::Io)? else {
        return Ok(Value::null());
    };
    let heap = lana_vm::heap::Heap::new(256 * 1024 * 1024);
    let mut map = lana_vm::value::Map::new(&heap, names.len())?;
    for (index, name) in names.iter().enumerate() {
        let value = match row.get_ref(index).map_err(|_| LanaError::Io)? {
            ValueRef::Null => Value::null(),
            ValueRef::Integer(value) => Value::number(value as f64),
            ValueRef::Real(value) => Value::number(value),
            ValueRef::Text(bytes) => Value::string(Arc::from(std::str::from_utf8(bytes).map_err(|_| LanaError::Parse)?)),
            ValueRef::Blob(_) => return Err(LanaError::UnsupportedValue),
        };
        map.set(Arc::from(name.as_str()), value, true)?;
    }
    Ok(Value::map(Arc::new(std::sync::Mutex::new(map))))
}

#[cfg(not(target_arch = "wasm32"))]
fn http_fetch(port: u16, path: &str) -> Result<Value, LanaError> {
    use std::io::{Read, Write};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream};
    use std::time::Duration;

    if path.is_empty() || path.starts_with('/') || path.contains("://")
        || path.split('/').any(|part| part == "..")
        || path.bytes().any(|byte| byte <= b' ' || byte == 0x7f)
    {
        return Err(LanaError::Parse);
    }
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    let timeout = Duration::from_secs(5);
    let mut socket = TcpStream::connect_timeout(&address, timeout).map_err(|_| LanaError::Io)?;
    socket.set_read_timeout(Some(timeout)).map_err(|_| LanaError::Io)?;
    socket.set_write_timeout(Some(timeout)).map_err(|_| LanaError::Io)?;
    write!(socket, "GET /{path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n").map_err(|_| LanaError::Io)?;
    let mut response = Vec::new();
    socket.take(64 * 1024 + 1).read_to_end(&mut response).map_err(|_| LanaError::Io)?;
    if response.len() > 64 * 1024 {
        return Err(LanaError::Limit);
    }
    let boundary = response.windows(4).position(|part| part == b"\r\n\r\n").ok_or(LanaError::Parse)?;
    let header = std::str::from_utf8(&response[..boundary]).map_err(|_| LanaError::Parse)?;
    let status = header.split("\r\n").next().and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok()).ok_or(LanaError::Parse)?;
    let body = std::str::from_utf8(&response[boundary + 4..]).map_err(|_| LanaError::Parse)?;
    let parsed = data::json_parse(body)?;
    let ValueKind::Map(map) = parsed.kind else { return Err(LanaError::Corruption) };
    let map = map.lock().unwrap();
    let Some(Value { kind: ValueKind::Bool(ok), .. }) = map.get("ok") else {
        return Err(LanaError::Corruption);
    };
    if !ok {
        let Some(Value { kind: ValueKind::Map(error), .. }) = map.get("error") else {
            return Err(LanaError::Corruption);
        };
        let error = error.lock().unwrap();
        let Some(Value { kind: ValueKind::String(code), .. }) = error.get("code") else {
            return Err(LanaError::Corruption);
        };
        return Err(match code.as_ref() {
            "LANA_ERR_NOT_FOUND" => LanaError::NotFound,
            "LANA_ERR_LIMIT" => LanaError::Limit,
            "LANA_ERR_PARSE" => LanaError::Parse,
            "LANA_ERR_UNSUPPORTED_OPERATION" => LanaError::UnsupportedOperation,
            _ => LanaError::Io,
        });
    }
    if !(200..300).contains(&status) {
        return Err(LanaError::Io);
    }
    map.get("evidence").cloned().ok_or(LanaError::Corruption)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(kind: AdapterKind, config: Option<&str>) -> AdapterOptions {
        AdapterOptions { schema_version: 1, kind, config: config.map(Arc::from) }
    }

    #[test]
    fn json_remains_available() {
        let json = adapter_load(&options(AdapterKind::Json, None)).unwrap();
        assert!(matches!(adapter_fetch(&json, r#"{"a":1}"#).unwrap().kind, ValueKind::Map(_)));
        assert_eq!(adapter_load(&AdapterOptions { schema_version: 0, ..options(AdapterKind::Csv, None) }).unwrap_err(), LanaError::InvalidState);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sqlite_reads_one_row_without_write_authority() {
        let path = std::env::temp_dir().join(format!("lana-sqlite-{}-{:?}.db", std::process::id(), std::thread::current().id()));
        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute_batch("CREATE TABLE readings(sensor TEXT, reading REAL); INSERT INTO readings VALUES ('s1', 0.8), ('s2', NULL);").unwrap();
        drop(db);
        let adapter = adapter_load(&options(AdapterKind::Sqlite, path.to_str())).unwrap();
        let row = adapter_fetch(&adapter, "SELECT sensor, reading FROM readings WHERE sensor='s1'").unwrap();
        let ValueKind::Map(map) = row.kind else { panic!("expected map") };
        assert_eq!(map.lock().unwrap().get("reading").unwrap().as_number(), 0.8);
        assert!(matches!(adapter_fetch(&adapter, "SELECT sensor FROM readings WHERE sensor='none'").unwrap().kind, ValueKind::Null));
        assert_eq!(adapter_fetch(&adapter, "DELETE FROM readings").unwrap_err(), LanaError::UnsupportedOperation);
        drop(adapter);
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn http_reads_local_envelope_and_maps_errors() {
        use std::io::{Read, Write};
        let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = server.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            for body in [r#"{"ok":true,"evidence":{"p":0.9}}"#, r#"{"ok":false,"error":{"code":"LANA_ERR_NOT_FOUND"}}"#] {
                let (mut stream, _) = server.accept().unwrap();
                let mut request = Vec::new();
                let mut byte = [0; 1];
                while !request.ends_with(b"\r\n\r\n") {
                    stream.read_exact(&mut byte).unwrap();
                    request.push(byte[0]);
                }
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
            }
        });
        let adapter = adapter_load(&options(AdapterKind::HttpJson, Some(&port.to_string()))).unwrap();
        let value = adapter_fetch(&adapter, "evidence/sensor-1").unwrap();
        let ValueKind::Map(map) = value.kind else { panic!("expected map") };
        assert_eq!(map.lock().unwrap().get("p").unwrap().as_number(), 0.9);
        assert_eq!(adapter_fetch(&adapter, "evidence/missing").unwrap_err(), LanaError::NotFound);
        assert_eq!(adapter_fetch(&adapter, "../escape").unwrap_err(), LanaError::Parse);
        handle.join().unwrap();
    }
}
