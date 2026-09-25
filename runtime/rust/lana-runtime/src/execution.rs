//! LIP-029's narrow, origin-bound webhook execution boundary.

use std::sync::Arc;
use std::io::Write;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use ring::rand::{SecureRandom, SystemRandom};

use lana_bytecode::LanaError;
use lana_vm::value::{Value, ValueKind};

use crate::codec;
use crate::sha256;
use crate::store::{store_close, store_commit, store_get, store_put, Store};

#[derive(Clone)]
pub struct ExecutionCapability {
    id: Arc<str>,
    origin: Arc<str>,
}
/// Host-only execution metadata. Its on-disk form is `LXE1 || nonce || AES-GCM`.
/// Neither source nor bytecode receives the origin or credential identifier.
#[derive(Clone)]
pub struct ExecutionConfig {
    pub capability: ExecutionCapability,
    pub credential_key_id: Arc<str>,
    pub ca_file: Option<Arc<str>>,
}

fn private_file(path: &Path) -> Result<(), LanaError> {
    #[cfg(unix)] {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(path).map_err(|_| LanaError::Io)?.permissions().mode();
        if mode & 0o077 != 0 { return Err(LanaError::Capability); }
    }
    Ok(())
}

fn create_private_file(path: &Path) -> Result<fs::File, LanaError> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)] {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).map_err(|_| LanaError::Io)
}

struct TemporaryCredentialFile(PathBuf);

impl TemporaryCredentialFile {
    fn create(credential: &str) -> Result<Self, LanaError> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| LanaError::Io)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!("lana-curl-{}-{nanos}", std::process::id()));
        let escaped = credential.replace('\\', "\\\\").replace('"', "\\\"");
        let mut file = create_private_file(&path)?;
        let owned = Self(path);
        file.write_all(format!("header = \"Authorization: {escaped}\"\n").as_bytes())
            .map_err(|_| LanaError::Io)?;
        Ok(owned)
    }
}

impl Drop for TemporaryCredentialFile {
    fn drop(&mut self) { let _ = fs::remove_file(&self.0); }
}

fn config_key(path: &Path) -> Result<LessSafeKey, LanaError> {
    private_file(path)?;
    let key = fs::read(path).map_err(|_| LanaError::Io)?;
    if key.len() != 32 { return Err(LanaError::Schema); }
    Ok(LessSafeKey::new(UnboundKey::new(&AES_256_GCM, &key).map_err(|_| LanaError::Schema)?))
}

impl ExecutionConfig {
    pub fn write(metadata: &Path, key_path: &Path, id: &str, origin: &str, credential_key_id: &str, ca_file: Option<&str>) -> Result<(), LanaError> {
        let capability = ExecutionCapability::new(Arc::<str>::from(id), Arc::<str>::from(origin))?;
        if metadata == key_path || credential_key_id.is_empty() || credential_key_id.chars().any(char::is_control) || ca_file.is_some_and(|path| path.chars().any(char::is_control)) { return Err(LanaError::Schema); }
        let rng = SystemRandom::new();
        let mut key = [0u8; 32]; let mut nonce = [0u8; 12];
        rng.fill(&mut key).map_err(|_| LanaError::Io)?;
        rng.fill(&mut nonce).map_err(|_| LanaError::Io)?;
        let plain = format!("{}\t{}\t{}\t{}", capability.id(), capability.origin(), credential_key_id, ca_file.unwrap_or(""));
        let mut sealed = plain.into_bytes();
        let cipher = LessSafeKey::new(UnboundKey::new(&AES_256_GCM, &key).map_err(|_| LanaError::Schema)?);
        cipher.seal_in_place_append_tag(Nonce::assume_unique_for_key(nonce), Aad::empty(), &mut sealed).map_err(|_| LanaError::Io)?;
        let mut output = b"LXE1".to_vec(); output.extend_from_slice(&nonce); output.extend_from_slice(&sealed);
        let mut key_created = false;
        let mut metadata_created = false;
        let result = (|| {
            {
                let mut key_file = create_private_file(key_path)?;
                key_created = true;
                key_file.write_all(&key).map_err(|_| LanaError::Io)?;
            }
            {
                let mut metadata_file = create_private_file(metadata)?;
                metadata_created = true;
                metadata_file.write_all(&output).map_err(|_| LanaError::Io)?;
            }
            Ok(())
        })();
        if result.is_err() {
            if metadata_created { let _ = fs::remove_file(metadata); }
            if key_created { let _ = fs::remove_file(key_path); }
        }
        result
    }

    pub fn load(metadata: &Path, key_path: &Path) -> Result<Self, LanaError> {
        private_file(metadata)?;
        let mut input = fs::read(metadata).map_err(|_| LanaError::Io)?;
        if input.len() < 4 + 12 + 16 || &input[..4] != b"LXE1" { return Err(LanaError::Schema); }
        let nonce: [u8; 12] = input[4..16].try_into().map_err(|_| LanaError::Schema)?;
        let plain = config_key(key_path)?.open_in_place(Nonce::assume_unique_for_key(nonce), Aad::empty(), &mut input[16..]).map_err(|_| LanaError::Capability)?;
        let text = std::str::from_utf8(plain).map_err(|_| LanaError::Schema)?;
        let mut fields = text.split('\t');
        let id = fields.next().ok_or(LanaError::Schema)?;
        let origin = fields.next().ok_or(LanaError::Schema)?;
        let credential_key_id = fields.next().ok_or(LanaError::Schema)?;
        let ca_file = fields.next().ok_or(LanaError::Schema)?;
        if fields.next().is_some() || credential_key_id.is_empty() || credential_key_id.chars().any(char::is_control) || ca_file.chars().any(char::is_control) { return Err(LanaError::Schema); }
        Ok(Self { capability: ExecutionCapability::new(Arc::<str>::from(id), Arc::<str>::from(origin))?, credential_key_id: Arc::from(credential_key_id), ca_file: if ca_file.is_empty() { None } else { Some(Arc::from(ca_file)) } })
    }
}

impl ExecutionCapability {
    pub fn new(id: impl Into<Arc<str>>, origin: impl Into<Arc<str>>) -> Result<Self, LanaError> {
        let id = id.into();
        let origin = origin.into();
        if id.is_empty() || id.chars().any(char::is_control) || !valid_origin(&origin) {
            return Err(LanaError::Schema);
        }
        Ok(Self { id, origin })
    }

    pub fn id(&self) -> &str { &self.id }
    pub fn origin(&self) -> &str { &self.origin }
}

fn valid_origin(origin: &str) -> bool {
    let Some(authority) = origin.strip_prefix("https://") else { return false; };
    let (host, port) = if authority.starts_with('[') {
        let Some((host, rest)) = authority.split_once(']') else { return false; };
        if host[1..].parse::<std::net::Ipv6Addr>().is_err() { return false; }
        (host, rest.strip_prefix(':').or_else(|| if rest.is_empty() { Some("") } else { None }))
    } else {
        let (host, port) = authority.split_once(':').unwrap_or((authority, ""));
        if host.is_empty() || !host.split('.').all(|label| !label.is_empty()
            && !label.starts_with('-') && !label.ends_with('-')
            && label.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')) { return false; }
        (host, Some(port))
    };
    !host.is_empty() && port.is_some_and(|port| (port.is_empty() && !authority.ends_with(':'))
        || (!port.is_empty() && port.bytes().all(|byte| byte.is_ascii_digit()) && port.parse::<u16>().is_ok_and(|port| port != 0)))
}

fn valid_path(path: &str) -> bool {
    path.starts_with('/') && !path.starts_with("//")
        && !path.chars().any(|c| c.is_control() || c.is_whitespace() || matches!(c, ':' | '?' | '#' | '\\'))
}

pub(crate) fn validate_plan(plan: &Value) -> Result<(Arc<str>, String), LanaError> {
    plan.check_resolved(256 * 1024 * 1024)?;
    let ValueKind::Map(map) = &plan.kind else { return Err(LanaError::Schema); };
    let (kind, path, payload) = {
        let map = map.lock().unwrap();
        (map.get("kind").cloned(), map.get("path").cloned(), map.get("payload").cloned())
    };
    if !matches!(kind, Some(Value { kind: ValueKind::String(ref kind), .. }) if kind.as_ref() == "webhook") { return Err(LanaError::Schema); }
    let Some(Value { kind: ValueKind::String(path), .. }) = path else { return Err(LanaError::Schema); };
    if !valid_path(&path) { return Err(LanaError::Capability); }
    let body = codec::encode_value(&payload.ok_or(LanaError::Schema)?)?;
    Ok((path, body))
}

#[derive(Clone)]
pub struct Authorization {
    pub decision_id: u64,
    pub capability_id: Arc<str>,
    pub plan_digest: Arc<str>,
    pub authorized: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiptStatus { Pending, Succeeded, Failed, Unknown }

impl ReceiptStatus {
    pub fn name(self) -> &'static str {
        match self { Self::Pending => "Pending", Self::Succeeded => "Succeeded", Self::Failed => "Failed", Self::Unknown => "Unknown" }
    }
}

pub trait WebhookTransport {
    /// `Ok(status)` means a complete HTTP response was received. An `Err`
    /// means delivery is ambiguous and therefore becomes `Unknown`.
    fn post_json(&self, origin: &str, path: &str, body: &str) -> Result<u16, LanaError>;
}

/// Native HTTPS transport used by the Rust CLI. It deliberately exposes no
/// caller-controlled headers, URL, certificate switch, or retry policy.
pub struct CurlTransport { pub credential: Option<Arc<str>>, pub ca_file: Option<Arc<str>> }

impl WebhookTransport for CurlTransport {
    fn post_json(&self, origin: &str, path: &str, body: &str) -> Result<u16, LanaError> {
        if !valid_origin(origin) || !valid_path(path) { return Err(LanaError::Capability); }
        let mut command = Command::new("curl");
        command.args(["--disable", "--globoff", "--silent", "--show-error", "--request", "POST", "--header", "Content-Type: application/json", "--output", if cfg!(windows) { "NUL" } else { "/dev/null" }, "--write-out", "%{http_code}", "--proto", "=https", "--tlsv1.2", "--max-time", "30", "--data-binary", "@-"]);
        if let Some(ca_file) = &self.ca_file {
            command.args(["--cacert", ca_file]);
        }
        let config_file = if let Some(credential) = &self.credential {
            if credential.is_empty() || credential.chars().any(char::is_control) { return Err(LanaError::Capability); }
            Some(TemporaryCredentialFile::create(credential)?)
        } else { None };
        if let Some(file) = &config_file { command.arg("--config").arg(&file.0); }
        let mut child = command
            .arg(format!("{origin}{path}"))
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null())
            .spawn().map_err(|_| LanaError::Io)?;
        let write_result = child.stdin.as_mut().ok_or(LanaError::Io).and_then(|stdin| stdin.write_all(body.as_bytes()).map_err(|_| LanaError::Io));
        if write_result.is_err() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(LanaError::Io);
        }
        let output = child.wait_with_output().map_err(|_| LanaError::Io)?;
        if !output.status.success() { return Err(LanaError::Io); }
        std::str::from_utf8(&output.stdout).ok().and_then(|s| s.parse::<u16>().ok()).ok_or(LanaError::Io)
    }
}

pub fn plan_digest(plan: &Value) -> Result<Arc<str>, LanaError> {
    validate_plan(plan)?;
    let encoded = codec::encode_value(plan)?;
    let digest = sha256::sha256(encoded.as_bytes());
    let mut text = String::with_capacity(64);
    for byte in digest { text.push_str(&format!("{byte:02x}")); }
    Ok(Arc::from(text))
}

fn receipt_key(capability: &ExecutionCapability, digest: &str) -> String {
    format!("execution-receipt/{}/{}", capability.id(), digest)
}

fn receipt_value(status: ReceiptStatus, authorization: &Authorization, capability: &ExecutionCapability, digest: &str, http_status: u16) -> Result<Value, LanaError> {
    let id = codec::encode_value(&Value::string(Arc::from(capability.id())))?;
    let receipt_id = codec::encode_value(&Value::string(Arc::from(receipt_key(capability, digest))))?;
    let transport_status = if status == ReceiptStatus::Unknown { "unknown" } else if status == ReceiptStatus::Pending { "pending" } else { "received" };
    Ok(Value::string(Arc::from(format!(
        "{{\"record_schema\":1,\"id\":{},\"kind\":\"execution_receipt\",\"transport_status\":\"{}\",\"domain_status\":\"{}\",\"payload\":{{\"http_status\":{}}},\"error\":null,\"evidence\":[],\"assumptions\":[],\"exactness\":\"{}\",\"metadata\":{{}},\"authorization_id\":{},\"capability_id\":{},\"http_status\":{},\"plan_digest\":\"{}\",\"status\":\"{}\"}}",
        receipt_id, transport_status, status.name(), http_status,
        if status == ReceiptStatus::Unknown { "unknown" } else { "exact" },
        authorization.decision_id, id, http_status, digest, status.name()
    ))))
}

pub fn read_receipt(store: &Store, capability: &ExecutionCapability, digest: &str) -> Result<Value, LanaError> {
    let saved = store_get(store, &receipt_key(capability, digest))?;
    crate::data::json_parse(&saved.as_string())
}

/// Performs at most one transport attempt for a committed plan. Duplicate execution is rejected before transport;
/// a received non-2xx response is Failed and ambiguous delivery is Unknown.
pub fn execute<T: WebhookTransport>(
    store: &mut Store,
    capability: &ExecutionCapability,
    authorization: &Authorization,
    plan: &Value,
    path: &str,
    transport: &T,
) -> Result<ReceiptStatus, LanaError> {
    store.ensure_clean()?;
    let (plan_path, body) = validate_plan(plan)?;
    let digest = plan_digest(plan)?;
    if !authorization.authorized || authorization.capability_id.as_ref() != capability.id()
        || authorization.plan_digest.as_ref() != digest.as_ref() || path != plan_path.as_ref() {
        return Err(LanaError::Capability);
    }
    let key = receipt_key(capability, &digest);
    match store_get(store, &key) {
        Ok(_) => return Err(LanaError::Conflict),
        Err(LanaError::NotFound) => {}
        Err(error) => return Err(error),
    }
    commit_receipt(store, &key, &receipt_value(ReceiptStatus::Pending, authorization, capability, &digest, 0)?)?;

    let (status, http_status) = match transport.post_json(capability.origin(), path, &body) {
        Ok(code) if (200..300).contains(&code) => (ReceiptStatus::Succeeded, code),
        Ok(code) => (ReceiptStatus::Failed, code),
        Err(_) => (ReceiptStatus::Unknown, 0),
    };
    commit_receipt(store, &key, &receipt_value(status, authorization, capability, &digest, http_status)?)?;
    Ok(status)
}

fn commit_receipt(store: &mut Store, key: &str, value: &Value) -> Result<(), LanaError> {
    let result = store_put(store, key, value).and_then(|_| store_commit(store).map(|_| ()));
    if result.is_err() {
        // Reopen and reconcile the committed receipt before any further action.
        let _ = store_close(store);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{store_open, StoreOptions};
    use lana_vm::value::Map;
    use std::sync::{atomic::{AtomicUsize, Ordering}, Mutex};

    static NEXT_STORE: AtomicUsize = AtomicUsize::new(0);

    struct Response(Result<u16, LanaError>);
    impl WebhookTransport for Response {
        fn post_json(&self, _: &str, _: &str, _: &str) -> Result<u16, LanaError> { self.0 }
    }

    fn store() -> Store {
        let path = std::env::temp_dir().join(format!(
            "lana_execution_{}_{}",
            std::process::id(),
            NEXT_STORE.fetch_add(1, Ordering::Relaxed),
        ));
        let _ = std::fs::remove_dir_all(&path);
        store_open(&StoreOptions { schema_version: 1, path: path.to_string_lossy().into_owned(), timeout_ms: 0 }).unwrap()
    }

    fn plan() -> Value {
        let heap = lana_vm::heap::Heap::new(1024 * 1024);
        let mut map = Map::new(&heap, 3).unwrap();
        map.set(Arc::from("kind"), Value::string(Arc::from("webhook")), false).unwrap();
        map.set(Arc::from("path"), Value::string(Arc::from("/events")), false).unwrap();
        map.set(Arc::from("payload"), Value::string(Arc::from("payload")), false).unwrap();
        Value::map(Arc::new(Mutex::new(map)))
    }

    #[test]
    fn receipt_is_bound_and_terminal_without_retry() {
        let mut store = store();
        let capability = ExecutionCapability::new("hook-a", "https://example.test").unwrap();
        let plan = plan();
        let digest = plan_digest(&plan).unwrap();
        let authorization = Authorization { decision_id: 7, capability_id: Arc::from("hook-a"), plan_digest: digest, authorized: true };
        assert_eq!(execute(&mut store, &capability, &authorization, &plan, "/events", &Response(Ok(204))).unwrap(), ReceiptStatus::Succeeded);
        assert_eq!(execute(&mut store, &capability, &authorization, &plan, "/events", &Response(Ok(204))).unwrap_err(), LanaError::Conflict);
    }

    #[test]
    fn non_response_is_unknown() {
        let mut store = store();
        let capability = ExecutionCapability::new("hook-b", "https://example.test").unwrap();
        let plan = plan();
        let authorization = Authorization { decision_id: 8, capability_id: Arc::from("hook-b"), plan_digest: plan_digest(&plan).unwrap(), authorized: true };
        assert_eq!(execute(&mut store, &capability, &authorization, &plan, "/events", &Response(Err(LanaError::Io))).unwrap(), ReceiptStatus::Unknown);
    }

    #[test]
    fn config_write_is_private_and_transactional() {
        let root = std::env::temp_dir().join(format!("lana_execution_config_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let metadata = root.join("metadata.lxe");
        let key = root.join("key");
        ExecutionConfig::write(&metadata, &key, "hook", "https://example.test", "credential", None).unwrap();
        #[cfg(unix)] {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(&metadata).unwrap().permissions().mode() & 0o777, 0o600);
            assert_eq!(fs::metadata(&key).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let missing_parent = root.join("missing").join("metadata.lxe");
        let rollback_key = root.join("rollback-key");
        assert_eq!(ExecutionConfig::write(&missing_parent, &rollback_key, "hook", "https://example.test", "credential", None), Err(LanaError::Io));
        assert!(!rollback_key.exists());
        assert_eq!(ExecutionConfig::write(&metadata, &metadata, "hook", "https://example.test", "credential", None), Err(LanaError::Schema));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn temporary_credentials_are_private_and_removed() {
        let file = TemporaryCredentialFile::create("Bearer test-token").unwrap();
        let path = file.0.clone();
        #[cfg(unix)] {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        drop(file);
        assert!(!path.exists());
    }

    #[test]
    fn invalid_plans_and_staged_writes_do_not_commit_or_send() {
        struct Never;
        impl WebhookTransport for Never {
            fn post_json(&self, _: &str, _: &str, _: &str) -> Result<u16, LanaError> { panic!("unexpected transport"); }
        }
        let mut store = store();
        let capability = ExecutionCapability::new("quoted-\"id", "https://example.test").unwrap();
        let plan = plan();
        let auth = Authorization { decision_id: 1, capability_id: Arc::from(capability.id()), plan_digest: plan_digest(&plan).unwrap(), authorized: true };
        assert_eq!(execute(&mut store, &capability, &auth, &plan, "/different", &Never), Err(LanaError::Capability));
        assert_eq!(execute(&mut store, &capability, &auth, &Value::null(), "/events", &Never), Err(LanaError::Schema));
        assert_eq!(crate::store::store_current_revision(&store).unwrap().revision_id, 0);
        store_put(&mut store, "unrelated", &Value::number(1.0)).unwrap();
        assert_eq!(execute(&mut store, &capability, &auth, &plan, "/events", &Never), Err(LanaError::InvalidState));
        assert_eq!(crate::store::store_current_revision(&store).unwrap().revision_id, 0);
        store_commit(&mut store).unwrap();
        assert_eq!(store_get(&store, "unrelated").unwrap().as_number(), 1.0);
        assert_eq!(execute(&mut store, &capability, &auth, &plan, "/events", &Response(Ok(302))).unwrap(), ReceiptStatus::Failed);
        let receipt = store_get(&store, &receipt_key(&capability, &auth.plan_digest)).unwrap();
        assert!(crate::data::json_parse(&receipt.as_string()).is_ok());
    }

    #[test]
    fn uncertain_commit_closes_store_and_reopen_blocks_repeat() {
        let mut store = store();
        let root = crate::store::store_get_path(&store).unwrap();
        let capability = ExecutionCapability::new("commit", "https://example.test").unwrap();
        let plan = plan();
        let auth = Authorization { decision_id: 1, capability_id: Arc::from("commit"), plan_digest: plan_digest(&plan).unwrap(), authorized: true };
        let manifest = Path::new(&root).join("manifest");
        let _ = fs::remove_file(&manifest);
        fs::create_dir(&manifest).unwrap();
        assert_eq!(execute(&mut store, &capability, &auth, &plan, "/events", &Response(Ok(204))), Err(LanaError::Io));
        assert_eq!(store.ensure_clean(), Err(LanaError::InvalidState));
        fs::remove_dir(&manifest).unwrap();
        let mut reopened = store_open(&StoreOptions { schema_version: 1, path: root, timeout_ms: 10 }).unwrap();
        assert_eq!(execute(&mut reopened, &capability, &auth, &plan, "/events", &Response(Ok(204))), Err(LanaError::Conflict));
    }

    #[test]
    fn origins_and_paths_cannot_widen_authority() {
        for origin in ["http://example.test", "https://", "https://a@b", "https://a/path", "https://a?b", "https://a#b", "https://a\\b", "https://a:0", "https://a:65536", "https://a:", "https://{a,b}", "https://a\n"] {
            assert!(ExecutionCapability::new("test", origin).is_err(), "{origin:?}");
        }
        for origin in ["https://example.test", "https://127.0.0.1:443", "https://[::1]:8443"] {
            assert!(ExecutionCapability::new("test", origin).is_ok(), "{origin}");
        }
        for path in ["//elsewhere", "/x\n", "/x y", "/x?y", "/x#y", "/x:y", "/x\\y"] { assert!(!valid_path(path)); }
    }
}
