//! LIP-029's narrow, origin-bound webhook execution boundary.

use std::sync::Arc;
use std::io::Write;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use ring::rand::{SecureRandom, SystemRandom};

use lana_bytecode::LanaError;
use lana_vm::value::Value;

use crate::codec;
use crate::sha256;
use crate::store::{store_commit, store_get, store_put, Store};

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
        let result = (|| {
            let mut file = create_private_file(&path)?;
            file.write_all(format!("header = \"Authorization: {escaped}\"\n").as_bytes())
                .map_err(|_| LanaError::Io)
        })();
        if result.is_err() { let _ = fs::remove_file(&path); }
        result.map(|_| Self(path))
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
        if metadata == key_path || credential_key_id.is_empty() || credential_key_id.contains(['\n', '\r', '\t']) { return Err(LanaError::Schema); }
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
        if fields.next().is_some() || credential_key_id.is_empty() { return Err(LanaError::Schema); }
        Ok(Self { capability: ExecutionCapability::new(Arc::<str>::from(id), Arc::<str>::from(origin))?, credential_key_id: Arc::from(credential_key_id), ca_file: if ca_file.is_empty() { None } else { Some(Arc::from(ca_file)) } })
    }
}

impl ExecutionCapability {
    pub fn new(id: impl Into<Arc<str>>, origin: impl Into<Arc<str>>) -> Result<Self, LanaError> {
        let id = id.into();
        let origin = origin.into();
        if id.is_empty() || !origin.starts_with("https://") || origin[8..].is_empty()
            || origin.contains('?') || origin.contains('#') || origin.ends_with('/') {
            return Err(LanaError::Schema);
        }
        Ok(Self { id, origin })
    }

    pub fn id(&self) -> &str { &self.id }
    pub fn origin(&self) -> &str { &self.origin }
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
        let mut command = Command::new("curl");
        command.args(["--silent", "--show-error", "--request", "POST", "--header", "Content-Type: application/json", "--output", "/dev/null", "--write-out", "%{http_code}", "--proto", "=https", "--tlsv1.2", "--max-time", "30", "--data-binary", "@-"]);
        if let Some(ca_file) = &self.ca_file {
            command.args(["--cacert", ca_file]);
        }
        let config_file = if let Some(credential) = &self.credential {
            if credential.is_empty() || credential.contains(['\r', '\n']) { return Err(LanaError::Capability); }
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
    let encoded = codec::encode_value(plan)?;
    let digest = sha256::sha256(encoded.as_bytes());
    let mut text = String::with_capacity(64);
    for byte in digest { text.push_str(&format!("{byte:02x}")); }
    Ok(Arc::from(text))
}

fn receipt_key(capability: &ExecutionCapability, digest: &str) -> String {
    format!("execution-receipt/{}/{}", capability.id(), digest)
}

fn receipt_value(status: ReceiptStatus, authorization: &Authorization, capability: &ExecutionCapability, digest: &str, http_status: u16) -> Value {
    Value::string(Arc::from(format!(
        "{{\"authorization_id\":{},\"capability_id\":\"{}\",\"http_status\":{},\"plan_digest\":\"{}\",\"record_schema\":1,\"status\":\"{}\"}}",
        authorization.decision_id, capability.id(), http_status, digest, status.name()
    )))
}

/// Executes exactly once. Duplicate execution is rejected before transport;
/// a received non-2xx response is Failed and ambiguous delivery is Unknown.
pub fn execute<T: WebhookTransport>(
    store: &mut Store,
    capability: &ExecutionCapability,
    authorization: &Authorization,
    plan: &Value,
    path: &str,
    transport: &T,
) -> Result<ReceiptStatus, LanaError> {
    let digest = plan_digest(plan)?;
    if !authorization.authorized || authorization.capability_id.as_ref() != capability.id()
        || authorization.plan_digest.as_ref() != digest.as_ref() || path.is_empty() || !path.starts_with('/')
        || path.contains("://") || path.contains('?') || path.contains('#') || path.contains('\\') {
        return Err(LanaError::Capability);
    }
    let key = receipt_key(capability, &digest);
    match store_get(store, &key) {
        Ok(_) => return Err(LanaError::Conflict),
        Err(LanaError::NotFound) => {}
        Err(error) => return Err(error),
    }
    store_put(store, &key, &receipt_value(ReceiptStatus::Pending, authorization, capability, &digest, 0))?;
    store_commit(store)?;

    let body = match &plan.kind {
        lana_vm::value::ValueKind::Map(map) => {
            let map = map.lock().unwrap();
            let Some(payload) = map.get("payload") else { return Err(LanaError::Schema); };
            codec::encode_value(payload)?
        }
        _ => return Err(LanaError::Schema),
    };
    let (status, http_status) = match transport.post_json(capability.origin(), path, &body) {
        Ok(code) if (200..300).contains(&code) => (ReceiptStatus::Succeeded, code),
        Ok(code) => (ReceiptStatus::Failed, code),
        Err(_) => (ReceiptStatus::Unknown, 0),
    };
    store_put(store, &key, &receipt_value(status, authorization, capability, &digest, http_status))?;
    store_commit(store)?;
    Ok(status)
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
        let mut map = Map::new(&heap, 1).unwrap();
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
}
