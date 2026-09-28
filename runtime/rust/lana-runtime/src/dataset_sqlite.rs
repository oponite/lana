//! One bounded, read-only SQLite snapshot with typed source evidence.

use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

use lana_bytecode::LanaError;
use lana_vm::heap::Heap;
use lana_vm::value::{Array, Map, Value, ValueKind};
use lana_vm::Vm;
use rusqlite::hooks::{AuthAction, Authorization};
use rusqlite::types::{Value as SqlValue, ValueRef};
use rusqlite::{params_from_iter, Connection, OpenFlags};
use serde_json::json;

use crate::information_codec::{canonical, Tagged};

const MAX_ROWS: usize = 10_000;
const MAX_COLUMNS: usize = 64;
const MAX_SQL: usize = 16 * 1024 * 1024;
const MAX_OUTPUT: usize = 64 * 1024 * 1024;
const MAX_EXACT_INTEGER: i64 = 1_i64 << 53;

fn digest(bytes: &[u8]) -> String {
    crate::sha256::sha256(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn map(heap: &Heap, fields: Vec<(String, Value)>) -> Result<Value, LanaError> {
    let mut result = Map::new(heap, fields.len())?;
    for (name, value) in fields { result.set(Arc::from(name), value, false)?; }
    Ok(Value::map(Arc::new(Mutex::new(result))))
}

fn array(heap: &Heap, values: Vec<Value>) -> Result<Value, LanaError> {
    Ok(Value::array(Arc::new(Mutex::new(Array::from_items(heap, values)?))))
}

fn string(value: impl Into<String>) -> Value {
    Value::string(Arc::from(value.into()))
}

fn parameter(value: &Value) -> Result<(SqlValue, Tagged), LanaError> {
    if value.reactive.is_some() || value.planned_effect.is_some() { return Err(LanaError::UnsupportedValue); }
    let sql = match &value.kind {
        ValueKind::Null => SqlValue::Null,
        ValueKind::Bool(value) => SqlValue::Integer(i64::from(*value)),
        ValueKind::Number(value) if value.is_finite() => SqlValue::Real(*value),
        ValueKind::String(value) => SqlValue::Text(value.to_string()),
        _ => return Err(LanaError::UnsupportedValue),
    };
    Ok((sql, Tagged::plain(value, 0)?))
}

fn cell(raw: ValueRef<'_>, kind: &str, vm: &mut Vm) -> Result<(Value, Tagged), LanaError> {
    let nullable = kind.starts_with("nullable_");
    if matches!(raw, ValueRef::Null) {
        return if nullable { Ok((Value::null(), Tagged::Null)) } else { Err(LanaError::Schema) };
    }
    let basic = kind.strip_prefix("nullable_").unwrap_or(kind);
    let value = match (basic, raw) {
        ("bool", ValueRef::Integer(value @ 0..=1)) => Value::boolean(value == 1),
        ("number", ValueRef::Integer(value)) if (-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&value) => Value::number(value as f64),
        ("number", ValueRef::Real(value)) if value.is_finite() => Value::number(value),
        ("string", ValueRef::Text(bytes)) => string(std::str::from_utf8(bytes).map_err(|_| LanaError::Schema)?),
        ("information_json", ValueRef::Text(bytes)) => {
            let tagged: Tagged = serde_json::from_slice(bytes).map_err(|_| LanaError::Schema)?;
            let encoded = canonical(&tagged)?;
            if encoded != bytes { return Err(LanaError::Schema); }
            tagged.to_live(vm)?;
            return Ok((string(std::str::from_utf8(&encoded).map_err(|_| LanaError::Schema)?), tagged));
        }
        _ => return Err(LanaError::Schema),
    };
    let tagged = Tagged::plain(&value, 0)?;
    Ok((value, tagged))
}

pub fn execute(path: &str, sql: &str, parameters: &[Value], schema: &[(String, String)], heap: &Heap, vm: &mut Vm) -> Result<Value, LanaError> {
    if sql.len() > MAX_SQL || schema.is_empty() || schema.len() > MAX_COLUMNS { return Err(LanaError::Limit); }
    if sql.contains('\0') { return Err(LanaError::Parse); }
    if !schema.iter().any(|(name, kind)| name == "id" && kind == "string") { return Err(LanaError::Schema); }
    let mut names = HashSet::new();
    for (name, kind) in schema {
        if name.is_empty() || !names.insert(name.as_str()) || !matches!(kind.as_str(), "bool" | "number" | "string" | "nullable_bool" | "nullable_number" | "nullable_string" | "information_json") {
            return Err(LanaError::Schema);
        }
    }
    let (bound, tagged_parameters): (Vec<_>, Vec<_>) = parameters.iter().map(parameter).collect::<Result<Vec<_>, _>>()?.into_iter().unzip();
    let parameter_bytes = canonical(&tagged_parameters)?;
    if parameter_bytes.len() > MAX_OUTPUT { return Err(LanaError::Limit); }

    let connection = Connection::open_with_flags(Path::new(path), OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|_| LanaError::Io)?;
    connection.execute_batch("BEGIN DEFERRED").map_err(|_| LanaError::Io)?;
    // Reading the catalog first pins the WAL snapshot before the user query runs.
    let ordinary_tables: HashSet<String> = {
        let mut catalog = connection.prepare("SELECT name FROM sqlite_schema WHERE type = 'table' AND sql NOT LIKE 'CREATE VIRTUAL TABLE%'").map_err(|_| LanaError::Io)?;
        let tables = catalog.query_map([], |row| row.get(0)).map_err(|_| LanaError::Io)?
            .collect::<Result<_, _>>().map_err(|_| LanaError::Io)?;
        tables
    };
    let builtin_functions: HashSet<String> = {
        let mut catalog = connection.prepare("PRAGMA function_list").map_err(|_| LanaError::Io)?;
        let functions = catalog.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))).map_err(|_| LanaError::Io)?
            .collect::<Result<Vec<_>, _>>().map_err(|_| LanaError::Io)?;
        functions.into_iter().filter(|(name, builtin)| *builtin == 1 && name != "load_extension" && name != "sqlite_log")
            .map(|(name, _)| name).collect()
    };
    connection.authorizer(Some(move |context: rusqlite::hooks::AuthContext<'_>| match context.action {
        AuthAction::Select | AuthAction::Recursive => Authorization::Allow,
        AuthAction::Read { table_name, .. } if context.database_name == Some("main") && ordinary_tables.contains(table_name) => Authorization::Allow,
        AuthAction::Function { function_name } if builtin_functions.contains(&function_name.to_ascii_lowercase()) => Authorization::Allow,
        _ => Authorization::Deny,
    }));
    let mut statement = connection.prepare(sql).map_err(|_| LanaError::UnsupportedOperation)?;
    if !statement.readonly() || statement.column_count() == 0 || statement.column_count() != schema.len() || statement.parameter_count() != bound.len() { return Err(LanaError::Schema); }
    let columns: Vec<String> = (0..statement.column_count()).map(|index| statement.column_name(index).map(str::to_string).map_err(|_| LanaError::Schema)).collect::<Result<_, _>>()?;
    if columns.iter().zip(schema).any(|(column, (name, _))| column != name) { return Err(LanaError::Schema); }

    let mut rows = statement.query(params_from_iter(bound.iter())).map_err(|_| LanaError::Io)?;
    let mut output_rows = Vec::new();
    let mut tagged_rows = Vec::new();
    let mut row_ids = HashSet::new();
    while let Some(row) = rows.next().map_err(|_| LanaError::Io)? {
        if output_rows.len() == MAX_ROWS { return Err(LanaError::Limit); }
        let mut fields = Vec::with_capacity(schema.len());
        let mut tagged = Vec::with_capacity(schema.len());
        for (index, (name, kind)) in schema.iter().enumerate() {
            let (value, snapshot) = cell(row.get_ref(index).map_err(|_| LanaError::Io)?, kind, vm)?;
            fields.push((name.clone(), value));
            tagged.push(snapshot);
        }
        let Some((_, Value { kind: ValueKind::String(id), .. })) = fields.iter().find(|(name, _)| name == "id") else { return Err(LanaError::Schema); };
        if id.is_empty() || id.len() > 128 || !row_ids.insert(id.to_string()) { return Err(LanaError::Schema); }
        output_rows.push(fields);
        tagged_rows.push(tagged);
    }
    drop(rows);
    drop(statement);
    let preimage = canonical(&json!({"columns": columns, "parameters": tagged_parameters, "rows": tagged_rows, "schema": schema.iter().map(|(name, kind)| json!({"name":name,"kind":kind})).collect::<Vec<_>>()}))?;
    if preimage.len() > MAX_OUTPUT { return Err(LanaError::Limit); }
    let source_revision = digest(&preimage);
    let sql_digest = digest(sql.as_bytes());
    let parameter_digest = digest(&parameter_bytes);
    let kinds = || array(heap, schema.iter().map(|(_, kind)| string(kind.clone())).collect());
    let mut result_rows = Vec::with_capacity(output_rows.len());
    let mut evidence = Vec::with_capacity(output_rows.len());
    for fields in output_rows {
        let row_id = match &fields.iter().find(|(name, _)| name == "id").unwrap().1.kind { ValueKind::String(id) => id.to_string(), _ => unreachable!() };
        result_rows.push(map(heap, fields)?);
        evidence.push(map(heap, vec![
            ("path".into(), string(path)), ("sql_digest".into(), string(&sql_digest)),
            ("parameter_digest".into(), string(&parameter_digest)),
            ("source_revision".into(), string(&source_revision)), ("row_id".into(), string(row_id)),
            ("column_types".into(), kinds()?),
        ])?);
    }
    connection.authorizer(None::<fn(rusqlite::hooks::AuthContext<'_>) -> Authorization>);
    connection.execute_batch("COMMIT").map_err(|_| LanaError::Io)?;
    let output = map(heap, vec![("rows".into(), array(heap, result_rows)?),
        ("source_revision".into(), string(source_revision)), ("evidence".into(), array(heap, evidence)?)])?;
    if canonical(&Tagged::plain(&output, 0)?)?.len() > MAX_OUTPUT { return Err(LanaError::Limit); }
    Ok(output)
}
