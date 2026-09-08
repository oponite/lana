//! Pipe-driven public-API probe for tests/test_store_process.py.
use std::io::{self, BufRead, Write};
use lana_bytecode::LanaError;
use lana_runtime::store::*;
use lana_vm::{Value, ValueKind};

fn run() -> Result<(), LanaError> {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() == 4 && args[1] == "--probe");
    println!("ATTEMPT");
    io::stdout().flush().map_err(|_| LanaError::Io)?;
    let mut store = store_open(&StoreOptions {
        schema_version: 1, path: args[2].clone(), timeout_ms: args[3].parse().unwrap(),
    })?;
    println!("OPEN");
    io::stdout().flush().map_err(|_| LanaError::Io)?;
    for command in io::stdin().lock().lines() {
        let command = command.map_err(|_| LanaError::Io)?;
        if command == "exit" { break; }
        let result = if let Some(number) = command.strip_prefix("put ") {
            store_put(&mut store, "key", &Value::number(number.parse().unwrap())).map(|_| "OK".into())
        } else if let Some(path) = command.strip_prefix("open ") {
            if store_current_revision(&store).is_ok() {
                Err(LanaError::InvalidState)
            } else {
                store_open(&StoreOptions { schema_version: 1, path: path.into(), timeout_ms: 1000 })
                    .map(|opened| { store = opened; "OPEN".into() })
            }
        } else {
            match command.as_str() {
                "stage" => store_put(&mut store, "staged", &Value::number(99.0)).map(|_| "OK".into()),
                "get" | "staged" => store_get(&store, if command == "get" { "key" } else { "staged" }).map(|v| {
                    let ValueKind::Number(number) = v.kind else { panic!("numeric fixture expected") };
                    format!("VALUE {number:.0}")
                }),
                "commit" => store_commit(&mut store).map(|r| format!("REV {}", r.revision_id)),
                "snapshot" => store_snapshot(&mut store).map(|(_, r)| format!("REV {}", r.revision_id)),
                "compact" => store_compact(&mut store, 0).map(|r| format!("REV {}", r.revision_id)),
                "close" => store_close(&mut store).map(|_| "OK".into()),
                _ => Err(LanaError::InvalidState),
            }
        };
        println!("{}", result.unwrap_or_else(|error| format!("ERR {}", error.name())));
        io::stdout().flush().map_err(|_| LanaError::Io)?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        println!("ERR {}", error.name());
        std::process::exit(1);
    }
}
