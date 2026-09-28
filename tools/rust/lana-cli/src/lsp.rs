//! Workspace queries use compiler declarations and byte spans, converted at the protocol boundary.
use super::{find_compiler, run_compiler_program, temp_path, CliError, LANA_VERSION};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::{BTreeMap}, io::{BufRead, Read, Write}, path::{Path, PathBuf}, process::ExitCode};

#[derive(Clone, Deserialize)]
struct Symbol {
    identity: String, name: String, path: String, line: usize, column: usize,
    #[serde(default)] kind: String,
    #[serde(default, rename = "type")] value_type: String,
}
#[derive(Default, Deserialize)]
struct Symbols { definitions: Vec<Symbol>, references: Vec<Symbol>, #[serde(default)] reserved: Vec<String> }
#[derive(Default)]
struct Workspace { roots: Vec<PathBuf>, documents: BTreeMap<PathBuf, String>, uris: BTreeMap<PathBuf, String> }

fn canonical(path: &Path) -> Result<PathBuf, String> {
    if let Ok(path) = path.canonicalize() { return Ok(path); }
    let parent = path.parent().ok_or("file has no parent")?.canonicalize().map_err(|e| e.to_string())?;
    Ok(parent.join(path.file_name().ok_or("file has no name")?))
}
fn path(uri: &str) -> Result<PathBuf, String> {
    let raw = uri.strip_prefix("file://").ok_or("only file URIs are supported")?;
    if !raw.starts_with('/') { return Err("file URI must have an absolute local path".into()); }
    let mut bytes = Vec::new(); let mut i = 0;
    while i < raw.len() {
        if raw.as_bytes()[i] == b'%' {
            let hex = raw.get(i + 1..i + 3).ok_or("invalid URI escape")?;
            bytes.push(u8::from_str_radix(hex, 16).map_err(|_| "invalid URI escape")?); i += 3;
        } else { bytes.push(raw.as_bytes()[i]); i += 1; }
    }
    let text = String::from_utf8(bytes).map_err(|_| "URI is not UTF-8")?;
    if text.contains('\0') { return Err("invalid URI path".into()); }
    canonical(Path::new(&text))
}
fn uri(path: &Path) -> String {
    let mut result = String::from("file://");
    for byte in path.to_string_lossy().bytes() {
        if byte.is_ascii_alphanumeric() || b"/-._~".contains(&byte) { result.push(byte as char); }
        else { result.push_str(&format!("%{byte:02X}")); }
    }
    result
}
fn source<'a>(sources: &'a BTreeMap<PathBuf, String>, symbol: &Symbol) -> Result<&'a str, String> {
    sources.get(Path::new(&symbol.path)).map(String::as_str).ok_or_else(|| "symbol source is unavailable".into())
}
fn byte_span(text: &str, symbol: &Symbol) -> Result<(usize, usize), String> {
    let line = text.split('\n').nth(symbol.line.checked_sub(1).ok_or("invalid symbol line")?).ok_or("invalid symbol line")?;
    let start = symbol.column.checked_sub(1).ok_or("invalid symbol column")?;
    let end = start.checked_add(symbol.name.len()).ok_or("invalid symbol span")?;
    if line.get(start..end) != Some(&symbol.name) { return Err(format!("compiler source span disagrees for {}", symbol.name)); }
    Ok((start, end))
}
fn range(text: &str, symbol: &Symbol) -> Result<Value, String> {
    let (start, end) = byte_span(text, symbol)?;
    let line = text.split('\n').nth(symbol.line - 1).unwrap();
    Ok(json!({"start":{"line":symbol.line-1,"character":line[..start].encode_utf16().count()},
        "end":{"line":symbol.line-1,"character":line[..end].encode_utf16().count()}}))
}
fn location(sources: &BTreeMap<PathBuf, String>, symbol: &Symbol) -> Result<Value, String> {
    Ok(json!({"uri":uri(Path::new(&symbol.path)),"range":range(source(sources, symbol)?, symbol)?}))
}
fn analyze(compiler: &Path, path: &Path, sources: &BTreeMap<PathBuf, String>) -> Result<Symbols, String> {
    let input = temp_path("lana-lsp-source"); let output = temp_path("lana-lsp-symbols"); let overlays = temp_path("lana-lsp-overlays");
    let result = (|| {
        std::fs::write(&input, sources.get(path).ok_or("source unavailable")?).map_err(|e| e.to_string())?;
        std::fs::write(&overlays, serde_json::to_vec(sources).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let args = vec!["--symbols".into(), input.to_string_lossy().into_owned(), output.to_string_lossy().into_owned(),
            path.to_string_lossy().into_owned(), overlays.to_string_lossy().into_owned()];
        run_compiler_program(compiler, &args).map_err(|e| match e { CliError::Run(info) => info.message, _ => "compiler analysis failed".into() })?;
        serde_json::from_slice(&std::fs::read(&output).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
    })();
    for file in [input, output, overlays] { let _ = std::fs::remove_file(file); }
    result
}
fn scan(directory: &Path, sources: &mut BTreeMap<PathBuf, String>) -> Result<(), String> {
    for item in std::fs::read_dir(directory).map_err(|e| e.to_string())? {
        let item = item.map_err(|e| e.to_string())?;
        let kind = item.file_type().map_err(|e| e.to_string())?;
        if kind.is_symlink() { continue; }
        let path = item.path();
        if kind.is_dir() {
            if !matches!(item.file_name().to_str(), Some(".git" | ".lana" | "target" | "node_modules" | ".venv")) { scan(&path, sources)?; }
        } else if path.extension().is_some_and(|ext| ext == "lana") {
            if sources.len() >= 4096 { return Err("workspace exceeds 4096 source files".into()); }
            sources.insert(canonical(&path)?, std::fs::read_to_string(path).map_err(|e| e.to_string())?);
        }
    }
    Ok(())
}
impl Workspace {
    fn client_result(&self, value: Value) -> Value {
        match value {
            Value::String(text) if text.starts_with("file://") => Value::String(path(&text).ok().and_then(|p| self.uris.get(&p).cloned()).unwrap_or(text)),
            Value::Array(values) => Value::Array(values.into_iter().map(|v| self.client_result(v)).collect()),
            Value::Object(values) => Value::Object(values.into_iter().map(|(key,value)| {
                let key = if key.starts_with("file://") { path(&key).ok().and_then(|p| self.uris.get(&p).cloned()).unwrap_or(key) } else {key};
                (key,self.client_result(value))
            }).collect()),
            value => value,
        }
    }
    fn sources(&self) -> Result<BTreeMap<PathBuf, String>, String> {
        let mut sources = BTreeMap::new();
        for root in &self.roots { scan(root, &mut sources)?; }
        sources.extend(self.documents.clone()); Ok(sources)
    }
    fn editable(&self, path: &Path) -> bool {
        !path.components().any(|part| part.as_os_str() == ".lana") &&
            (self.roots.iter().any(|root| path.starts_with(root)) || (self.roots.is_empty() && self.documents.contains_key(path)))
    }
    fn query(&self, compiler: &Path, method: &str, params: &Value) -> Result<Value, String> {
        let requested = path(params["textDocument"]["uri"].as_str().ok_or("missing URI")?)?;
        let mut sources = self.sources()?;
        if !sources.contains_key(&requested) { sources.insert(requested.clone(), std::fs::read_to_string(&requested).map_err(|e| e.to_string())?); }
        let mut all = Symbols::default();
        let workspace_query = matches!(method, "textDocument/references" | "textDocument/rename" | "textDocument/prepareRename");
        let paths: Vec<_> = if workspace_query { sources.keys().cloned().collect() } else { vec![requested.clone()] };
        for file in paths {
            let symbols = analyze(compiler, &file, &sources)?;
            all.definitions.extend(symbols.definitions); all.references.extend(symbols.references); all.reserved.extend(symbols.reserved);
        }
        for symbol in all.definitions.iter().chain(&all.references) {
            let path = PathBuf::from(&symbol.path);
            if !sources.contains_key(&path) { sources.insert(path.clone(), std::fs::read_to_string(path).map_err(|e| e.to_string())?); }
        }
        for list in [&mut all.definitions, &mut all.references] {
            list.sort_by(|a,b| (&a.path,a.line,a.column,&a.identity).cmp(&(&b.path,b.line,b.column,&b.identity)));
            list.dedup_by(|a,b| a.path == b.path && a.line == b.line && a.column == b.column && a.identity == b.identity);
            for symbol in list.iter() { byte_span(source(&sources, symbol)?, symbol)?; }
        }
        if method == "textDocument/completion" {
            let items: Vec<_> = all.definitions.iter().filter(|s| Path::new(&s.path) == requested)
                .map(|s| json!({"label":s.name,"kind":if s.kind=="function" {3} else {6},"detail":s.value_type})).collect();
            return Ok(json!({"isIncomplete":false,"items":items}));
        }
        let line = params["position"]["line"].as_u64().ok_or("missing line")?;
        let character = params["position"]["character"].as_u64().ok_or("missing character")?;
        let mut selected = None;
        for symbol in all.definitions.iter().chain(&all.references).filter(|s| Path::new(&s.path) == requested && s.line as u64 == line+1) {
            let span = range(source(&sources, symbol)?, symbol)?;
            if character >= span["start"]["character"].as_u64().unwrap() && character < span["end"]["character"].as_u64().unwrap() { selected = Some(symbol); break; }
        }
        let Some(selected) = selected else { return Ok(Value::Null); };
        let definition = all.definitions.iter().find(|s| s.identity == selected.identity);
        if method == "textDocument/hover" { return Ok(definition.map(|s| json!({"contents":{"kind":"plaintext","value":format!("{}: {} ({})",s.name,s.value_type,s.kind)}})).unwrap_or(Value::Null)); }
        if method == "textDocument/definition" { return definition.map(|s| location(&sources,s).map(|value| json!([value]))).unwrap_or(Ok(json!([]))); }
        let references = all.references.iter().filter(|s| s.identity == selected.identity);
        if method == "textDocument/references" {
            let mut values = references.map(|s| location(&sources,s)).collect::<Result<Vec<_>,_>>()?;
            if params["context"]["includeDeclaration"].as_bool().unwrap_or(false) {
                if let Some(definition) = definition { values.push(location(&sources,definition)?); }
            }
            return Ok(json!(values));
        }
        let definition = definition.ok_or("declaration is unavailable; rename is incomplete")?;
        let affected: Vec<_> = all.definitions.iter().chain(&all.references).filter(|s| s.identity == selected.identity).collect();
        if affected.iter().any(|s| !self.editable(Path::new(&s.path))) { return Err("dependency source is read-only".into()); }
        if method == "textDocument/prepareRename" { return Ok(json!({"range":range(source(&sources,selected)?,selected)?,"placeholder":selected.name})); }
        if method != "textDocument/rename" { return Err("unsupported method".into()); }
        let name = params["newName"].as_str().ok_or("missing new name")?;
        if !identifier(name) || all.reserved.iter().any(|word| word == name) { return Err("new name must be a nonreserved Lana identifier".into()); }
        if all.definitions.iter().any(|s| s.name == name && s.identity != definition.identity) { return Err("new name collides with an existing declaration".into()); }
        let mut changes: BTreeMap<String, Vec<Value>> = BTreeMap::new();
        let mut replacements: BTreeMap<PathBuf, Vec<&Symbol>> = BTreeMap::new();
        for symbol in affected {
            changes.entry(uri(Path::new(&symbol.path))).or_default().push(json!({"range":range(source(&sources,symbol)?,symbol)?,"newText":name}));
            replacements.entry(PathBuf::from(&symbol.path)).or_default().push(symbol);
        }
        for (path, symbols) in &mut replacements {
            symbols.sort_by_key(|s| std::cmp::Reverse((s.line,s.column)));
            let text = sources.get_mut(path).unwrap();
            for symbol in symbols {
                let (start,end) = byte_span(text,symbol)?;
                let offset: usize = text.split_inclusive('\n').take(symbol.line-1).map(str::len).sum();
                text.replace_range(offset+start..offset+end,name);
            }
        }
        // Recheck every editable source against the complete proposed overlay before emitting any edit.
        let candidates: Vec<_> = sources.keys().filter(|path| self.editable(path)).cloned().collect();
        for path in candidates { analyze(compiler,&path,&sources)?; }
        Ok(json!({"changes":changes}))
    }
}
fn identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes.next().is_some_and(|b| b.is_ascii_alphabetic() || b == b'_') && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_') &&
        !["fn","let","if","else","while","for","in","return","break","continue","true","false","null","import","as","type","match","state","probability","async","await","yield","class","value","interface","new","self","Self","public","private","mutable","static","replace","copies","implements","effects","transform","with","observe","measure","fork","join","print"].contains(&name)
}
fn send(output: &mut impl Write, message: Value) -> std::io::Result<()> {
    let body = message.to_string(); write!(output,"Content-Length: {}\r\n\r\n{}",body.len(),body)?; output.flush()
}
fn diagnostic(error: &str, text: &str) -> Value {
    for prefix in ["parse error at line ","type error at line ","lex error at line "] {
        if let Some(rest) = error.strip_prefix(prefix) {
            if let Some((line,rest)) = rest.split_once(" column ") { if let Some((column,message)) = rest.split_once(": ") {
                if let (Ok(line),Ok(column)) = (line.parse::<usize>(),column.parse::<usize>()) {
                    let source = text.split('\n').nth(line.saturating_sub(1)).unwrap_or("");
                    let offset = column.saturating_sub(1).min(source.len());
                    let offset = (0..=offset).rev().find(|&offset| source.is_char_boundary(offset)).unwrap();
                    let character = source[..offset].encode_utf16().count();
                    return json!({"range":{"start":{"line":line.saturating_sub(1),"character":character},"end":{"line":line.saturating_sub(1),"character":character+1}},"severity":1,"message":message});
                }
            } }
        }
    }
    json!({"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}},"severity":1,"message":error})
}
pub(super) fn run() -> ExitCode {
    let Some(compiler) = find_compiler() else { eprintln!("native Lana compiler bytecode not found"); return ExitCode::FAILURE; };
    let mut input = std::io::BufReader::new(std::io::stdin()); let mut output = std::io::stdout();
    let mut workspace = Workspace::default(); let mut shutdown = false;
    loop {
        let mut length = None;
        loop {
            let mut header = String::new();
            match input.read_line(&mut header) { Ok(0) => return ExitCode::SUCCESS, Err(_) => return ExitCode::FAILURE, _ => {} }
            if header.trim().is_empty() { break; }
            if let Some((key,value)) = header.split_once(':') { if key.eq_ignore_ascii_case("Content-Length") { length = value.trim().parse::<usize>().ok(); } }
        }
        let Some(length) = length.filter(|&n| n <= 16*1024*1024) else { return ExitCode::FAILURE; };
        let mut body = vec![0;length]; if input.read_exact(&mut body).is_err() { return ExitCode::FAILURE; }
        let Ok(request) = serde_json::from_slice::<Value>(&body) else { continue; };
        let method = request["method"].as_str().unwrap_or(""); let params = &request["params"];
        if method == "exit" { return if shutdown {ExitCode::SUCCESS} else {ExitCode::FAILURE}; }
        if method == "initialize" {
            if let Some(folders) = params["workspaceFolders"].as_array() {
                for folder in folders { if let Some(raw) = folder["uri"].as_str() { if let Ok(root) = path(raw) { workspace.roots.push(root); } } }
            } else if let Some(raw) = params["rootUri"].as_str() { if let Ok(root) = path(raw) { workspace.roots.push(root); } }
            if send(&mut output,json!({"jsonrpc":"2.0","id":request["id"],"result":{"serverInfo":{"name":"lana-lsp","version":LANA_VERSION},"capabilities":{"positionEncoding":"utf-16","textDocumentSync":1,"hoverProvider":true,"completionProvider":{},"definitionProvider":true,"referencesProvider":true,"renameProvider":{"prepareProvider":true}}}})).is_err() {return ExitCode::FAILURE;} continue;
        }
        if method == "shutdown" { shutdown=true; if send(&mut output,json!({"jsonrpc":"2.0","id":request["id"],"result":null})).is_err() {return ExitCode::FAILURE;} continue; }
        if matches!(method,"textDocument/didOpen"|"textDocument/didChange"|"textDocument/didClose") {
            let raw = params["textDocument"]["uri"].as_str().unwrap_or("");
            if let Ok(file) = path(raw) {
                workspace.uris.insert(file.clone(),raw.into());
                let diagnostics = if method == "textDocument/didClose" { workspace.documents.remove(&file); vec![] } else {
                    let text = params["textDocument"]["text"].as_str().or_else(|| params["contentChanges"][0]["text"].as_str()).unwrap_or("");
                    workspace.documents.insert(file.clone(),text.into());
                    match analyze(&compiler,&file,&workspace.documents) { Ok(_) => vec![], Err(error) => vec![diagnostic(&error,text)] }
                };
                if send(&mut output,json!({"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{"uri":raw,"diagnostics":diagnostics}})).is_err() {return ExitCode::FAILURE;}
            }
            continue;
        }
        if request.get("id").is_none() {continue;}
        let response = match workspace.query(&compiler,method,params) {
            Ok(value) => json!({"jsonrpc":"2.0","id":request["id"],"result":workspace.client_result(value)}),
            Err(error) => json!({"jsonrpc":"2.0","id":request["id"],"error":{"code":-32602,"message":error}}),
        };
        if send(&mut output,response).is_err() {return ExitCode::FAILURE;}
    }
}
