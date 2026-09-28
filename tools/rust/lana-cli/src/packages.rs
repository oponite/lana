//! Exact hosted source packages. Download never executes package code.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use flate2::{Compression, GzBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const COMPRESSED: usize = 64 * 1024 * 1024;
const EXPANDED: usize = 256 * 1024 * 1024;
static NEXT: AtomicU64 = AtomicU64::new(0);
type Result<T> = std::result::Result<T, String>;
fn invalid(message: &str) -> String { format!("LANA_ERR_SCHEMA: {message}") }
fn io(error: std::io::Error) -> String { format!("LANA_ERR_IO: {error}") }
fn sha(bytes: &[u8]) -> String { lana_runtime::sha256(bytes).iter().map(|b| format!("{b:02x}")).collect() }
fn bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path).map_err(io)?.take(limit as u64 + 1).read_to_end(&mut bytes).map_err(io)?;
    if bytes.len() > limit { return Err("LANA_ERR_LIMIT".into()); }
    Ok(bytes)
}
fn canonical<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = lana_runtime::information_codec::canonical(value).map_err(|e| e.name().to_string())?;
    bytes.push(b'\n');
    Ok(bytes)
}
fn publish(path: &Path, bytes: &[u8]) -> Result<()> {
    lana_runtime::atomic_file::write(path, bytes).map_err(|error| {
        if lana_runtime::atomic_file::durability_uncertain(&error) {
            format!("LANA_ERR_IO: durability uncertain; reload {}", path.display())
        } else { io(error) }
    })
}
fn name(value: &str) -> bool {
    !value.is_empty() && value.len() <= 100 && value != "." && value != ".." &&
        value.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"-_.".contains(&b))
}
fn version(value: &str) -> bool {
    let parts = value.split('.').collect::<Vec<_>>();
    parts.len() == 3 && parts.iter().all(|part| !part.is_empty() &&
        part.bytes().all(|b| b.is_ascii_digit()) && part.parse::<u64>().is_ok() &&
        (part.len() == 1 || !part.starts_with('0')))
}
fn identity(value: &str) -> bool {
    let parts = value.split('/').collect::<Vec<_>>();
    parts.len() == 2 && parts.iter().all(|part| name(part))
}
fn exact(value: &str) -> Result<(&str, &str)> {
    let (id, ver) = value.split_once('@').ok_or_else(|| invalid("expected owner/repo@X.Y.Z"))?;
    if !identity(id) || !version(ver) { return Err(invalid("invalid exact package identity")); }
    Ok((id, ver))
}
fn safe_path(value: &str) -> bool {
    !value.is_empty() && !value.contains(['\\', ':']) && !value.chars().any(char::is_control) &&
        value.split('/').all(|part| !part.is_empty() && part != "." && part != ".." &&
            !part.ends_with(['.', ' ']))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: u32, name: String, version: String, entry: String,
    #[serde(default)] hosted_dependencies: BTreeMap<String, String>,
    #[serde(default)] dependencies: BTreeMap<String, String>,
}
impl Manifest {
    fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 1024 * 1024 { return Err("LANA_ERR_LIMIT".into()); }
        let text = std::str::from_utf8(bytes).map_err(|_| invalid("manifest UTF-8"))?;
        let manifest: Self = toml::from_str(text).map_err(|e| invalid(&format!("manifest: {e}")))?;
        if manifest.schema != 1 || !name(&manifest.name) || !version(&manifest.version) ||
            !safe_path(&manifest.entry) || !manifest.entry.starts_with("src/") ||
            !manifest.entry.ends_with(".lana") || !manifest.dependencies.is_empty() {
            return Err(invalid("published manifest identity, entry, or local dependency"));
        }
        for (alias, dependency) in &manifest.hosted_dependencies {
            if !name(alias) { return Err(invalid("dependency alias")); }
            exact(dependency)?;
        }
        Ok(manifest)
    }
    fn root(&self) -> String { format!("{}-{}", self.name, self.version) }
    fn dependencies(&self) -> Vec<String> { self.hosted_dependencies.values().cloned().collect::<BTreeSet<_>>().into_iter().collect() }
}

struct Package { manifest: Manifest, entries: BTreeMap<String, Option<Vec<u8>>> }
fn walk(root: &Path, relative: &str, entries: &mut BTreeMap<String, Option<Vec<u8>>>, total: &mut usize) -> Result<()> {
    let path = root.join(relative);
    let metadata = fs::symlink_metadata(&path).map_err(io)?;
    if !safe_path(relative) || metadata.file_type().is_symlink() { return Err(invalid("unsafe package path")); }
    if metadata.is_dir() {
        entries.insert(relative.into(), None);
        if entries.len() > 2001 { return Err("LANA_ERR_LIMIT".into()); }
        for entry in fs::read_dir(&path).map_err(io)? {
            let entry = entry.map_err(io)?;
            let child = entry.file_name().into_string().map_err(|_| invalid("path UTF-8"))?;
            walk(root, &format!("{relative}/{child}"), entries, total)?;
        }
    } else if metadata.is_file() {
        let bytes = bounded(&path, EXPANDED.saturating_sub(*total))?;
        *total += bytes.len();
        entries.insert(relative.into(), Some(bytes));
        if entries.values().filter(|entry| entry.is_some()).count() > 1000 { return Err("LANA_ERR_LIMIT".into()); }
    } else { return Err(invalid("non-regular package file")); }
    Ok(())
}
fn from_directory(directory: &Path) -> Result<Package> {
    if fs::symlink_metadata(directory).map_err(io)?.file_type().is_symlink() { return Err(invalid("symlink package root")); }
    let mut entries = BTreeMap::new();
    let mut total = 0;
    for path in ["lana.toml", "src", "tests"] {
        if path == "tests" && fs::symlink_metadata(directory.join(path)).is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound) { continue; }
        walk(directory, path, &mut entries, &mut total)?;
    }
    let manifest = Manifest::parse(entries.get("lana.toml").and_then(Option::as_deref).ok_or_else(|| invalid("missing manifest"))?)?;
    if entries.get("src") != Some(&None) || !entries.get(&manifest.entry).is_some_and(Option::is_some) {
        return Err(invalid("missing src or entry"));
    }
    Ok(Package { manifest, entries })
}
fn archive(package: &Package) -> Result<Vec<u8>> {
    let mut builder = tar::Builder::new(Vec::new());
    let root = package.manifest.root();
    let mut entries = BTreeMap::from([(root.clone(), None)]);
    entries.extend(package.entries.iter().map(|(path, bytes)| (format!("{root}/{path}"), bytes.as_deref())));
    for (path, bytes) in entries {
        let mut header = tar::Header::new_ustar();
        header.set_path(if bytes.is_none() { format!("{path}/") } else { path }).map_err(io)?;
        header.set_entry_type(if bytes.is_some() { tar::EntryType::Regular } else { tar::EntryType::Directory });
        header.set_mode(if bytes.is_some() { 0o644 } else { 0o755 });
        header.set_uid(0); header.set_gid(0); header.set_mtime(0);
        header.set_size(bytes.map_or(0, |bytes| bytes.len()) as u64);
        header.set_cksum();
        builder.append(&header, bytes.unwrap_or(&[])).map_err(io)?;
    }
    let tar = builder.into_inner().map_err(io)?;
    if tar.len() > EXPANDED { return Err("LANA_ERR_LIMIT".into()); }
    let mut encoder = GzBuilder::new().mtime(0).operating_system(255).write(Vec::new(), Compression::new(6));
    encoder.write_all(&tar).map_err(io)?;
    let bytes = encoder.finish().map_err(io)?;
    if bytes.len() > COMPRESSED { return Err("LANA_ERR_LIMIT".into()); }
    Ok(bytes)
}
fn unpack(bytes: &[u8]) -> Result<Package> {
    if bytes.len() > COMPRESSED { return Err("LANA_ERR_LIMIT".into()); }
    let mut decoder = flate2::bufread::GzDecoder::new(std::io::Cursor::new(bytes));
    let mut expanded = Vec::new();
    (&mut decoder).take(EXPANDED as u64 + 1).read_to_end(&mut expanded).map_err(io)?;
    if expanded.len() > EXPANDED { return Err("LANA_ERR_LIMIT".into()); }
    if decoder.into_inner().position() != bytes.len() as u64 { return Err(invalid("trailing gzip data")); }
    let mut archive = tar::Archive::new(expanded.as_slice());
    let mut entries = BTreeMap::new();
    let mut names = BTreeSet::new();
    let mut end = 0;
    let mut file_count = 0;
    for entry in archive.entries().map_err(io)?.raw(true) {
        let mut entry = entry.map_err(io)?;
        let path = std::str::from_utf8(&entry.path_bytes()).map_err(|_| invalid("path UTF-8"))?.to_owned();
        let kind = entry.header().entry_type();
        if entry.header().as_ustar().is_none() || !(kind.is_file() || kind.is_dir()) { return Err(invalid("unsafe tar header or link")); }
        let path = if kind.is_dir() { path.strip_suffix('/').unwrap_or(&path) } else { &path }.to_owned();
        if !safe_path(&path) || !names.insert(path.to_lowercase()) { return Err(invalid("unsafe or duplicate archive path")); }
        let size = entry.size();
        if size > EXPANDED as u64 || (kind.is_dir() && size != 0) { return Err(invalid("invalid archive size")); }
        end = entry.raw_file_position().checked_add(size.div_ceil(512) * 512).ok_or_else(|| invalid("archive offset"))? as usize;
        let data = if kind.is_file() {
            file_count += 1;
            if file_count > 1000 { return Err("LANA_ERR_LIMIT".into()); }
            let mut data = Vec::new(); entry.read_to_end(&mut data).map_err(io)?;
            if data.len() as u64 != size { return Err(invalid("truncated archive file")); }
            Some(data)
        } else { None };
        entries.insert(path, data);
        if entries.len() > 2001 { return Err("LANA_ERR_LIMIT".into()); }
    }
    if end > expanded.len() || expanded.len().saturating_sub(end) < 1024 || expanded[end..].iter().any(|byte| *byte != 0) {
        return Err(invalid("missing terminator or trailing tar data"));
    }
    let roots = entries.keys().map(|path| path.split('/').next().unwrap()).collect::<BTreeSet<_>>();
    if roots.len() != 1 { return Err(invalid("one archive root required")); }
    let root = roots.first().unwrap().to_string();
    let manifest_path = format!("{root}/lana.toml");
    let manifest = Manifest::parse(entries.get(&manifest_path).and_then(Option::as_deref).ok_or_else(|| invalid("missing manifest"))?)?;
    if root != manifest.root() || entries.get(&root) != Some(&None) || entries.get(&format!("{root}/src")) != Some(&None) {
        return Err(invalid("archive root mismatch"));
    }
    entries.remove(&root);
    let entries = entries.into_iter().map(|(path, bytes)| (path[root.len()+1..].to_owned(), bytes)).collect::<BTreeMap<_, _>>();
    if entries.contains_key("tests") && entries.get("tests") != Some(&None) { return Err(invalid("tests must be a directory")); }
    for path in entries.keys() {
        if path != "lana.toml" && path != "src" && path != "tests" && !path.starts_with("src/") && !path.starts_with("tests/") {
            return Err(invalid("file outside package tree"));
        }
        if let Some((parent, _)) = path.rsplit_once('/') {
            if entries.get(parent) != Some(&None) { return Err(invalid("missing archive parent")); }
        }
    }
    if !entries.get(&manifest.entry).is_some_and(Option::is_some) { return Err(invalid("missing package entry")); }
    Ok(Package { manifest, entries })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Locked {
    identity: String, version: String, tag: String, asset: String, sha256: String,
    direct: bool, dependencies: Vec<String>,
}
impl Locked {
    fn exact(&self) -> String { format!("{}@{}", self.identity, self.version) }
    fn root(&self) -> String { format!("{}-{}", self.identity.split('/').nth(1).unwrap(), self.version) }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Lock { schema_version: u32, direct: Vec<String>, packages: Vec<Locked> }
impl Lock {
    fn empty() -> Self { Self { schema_version:1, direct:Vec::new(), packages:Vec::new() } }
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 || self.packages.len() > 64 || self.direct.is_empty() || !sorted(&self.direct) ||
            self.packages.windows(2).any(|pair| pair[0].identity >= pair[1].identity) { return Err(invalid("invalid lock schema/order")); }
        let mut visited = BTreeSet::new();
        for package in &self.packages {
            exact(&package.exact())?;
            if package.tag != format!("lana-v{}", package.version) || package.asset != format!("{}-lana.tar.gz", package.root()) ||
                package.sha256.len() != 64 || !package.sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) ||
                package.direct != self.direct.contains(&package.exact()) || !sorted(&package.dependencies) { return Err(invalid("invalid lock entry")); }
        }
        for direct in &self.direct { self.visit(direct, &mut BTreeSet::new(), &mut visited)?; }
        if visited.len() != self.packages.len() { return Err(invalid("unreachable lock entry")); }
        Ok(())
    }
    fn visit(&self, key: &str, stack: &mut BTreeSet<String>, visited: &mut BTreeSet<String>) -> Result<()> {
        exact(key)?;
        if stack.contains(key) { return Err(invalid("package cycle")); }
        if visited.contains(key) { return Ok(()); }
        let package = self.packages.iter().find(|package| package.exact() == key).ok_or_else(|| invalid("missing/conflicting locked dependency"))?;
        stack.insert(key.into());
        for dependency in &package.dependencies { self.visit(dependency, stack, visited)?; }
        stack.remove(key); visited.insert(key.into()); Ok(())
    }
}
fn sorted(values: &[String]) -> bool { values.windows(2).all(|pair| pair[0] < pair[1]) }
fn load_lock(project: &Path) -> Result<Option<Lock>> {
    let path = project.join("lana.lock");
    if !path.try_exists().map_err(io)? { return Ok(None); }
    let bytes = bounded(&path, 1024 * 1024)?;
    // Preserve the existing local-only lock format until the first explicit add.
    if bytes.starts_with(b"schema = 1\nproject = \"") { return Ok(None); }
    let lock: Lock = serde_json::from_slice(&bytes).map_err(|_| invalid("lock JSON"))?;
    if canonical(&lock)? != bytes { return Err(invalid("lock must be canonical JSON")); }
    lock.validate()?; Ok(Some(lock))
}

fn real_directory(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path).map_err(io)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() { return Err(invalid("cache directory is not owned regular storage")); }
    Ok(())
}
fn cache_parent(project: &Path, create: bool) -> Result<PathBuf> {
    let mut path = project.to_path_buf();
    for part in [".lana", "packages"] {
        path.push(part);
        if create && !path.try_exists().map_err(io)? { fs::create_dir(&path).map_err(io)?; }
        real_directory(&path)?;
    }
    Ok(path)
}
fn verify_cache(project: &Path, locked: &Locked) -> Result<PathBuf> {
    let directory = cache_parent(project, false)?.join(&locked.sha256);
    real_directory(&directory)?;
    let names = fs::read_dir(&directory).map_err(io)?.map(|entry| entry.map(|entry| entry.file_name()).map_err(io)).collect::<Result<BTreeSet<_>>>()?;
    if names != BTreeSet::from(["archive.tar.gz".into(), locked.root().into()]) { return Err(invalid("unexpected cache contents")); }
    let archive_path = directory.join("archive.tar.gz");
    if !fs::symlink_metadata(&archive_path).map_err(io)?.file_type().is_file() { return Err(invalid("archive is not a regular file")); }
    let bytes = bounded(&archive_path, COMPRESSED)?;
    if sha(&bytes) != locked.sha256 { return Err(invalid("cached archive digest mismatch")); }
    let package = unpack(&bytes)?;
    if package.manifest.root() != locked.root() || package.manifest.dependencies() != locked.dependencies {
        return Err(invalid("locked manifest mismatch"));
    }
    let root = directory.join(locked.root());
    real_directory(&root)?;
    let mut actual = BTreeMap::new(); let mut total = 0;
    for entry in fs::read_dir(&root).map_err(io)? {
        let file = entry.map_err(io)?.file_name().into_string().map_err(|_| invalid("cache path UTF-8"))?;
        walk(&root, &file, &mut actual, &mut total)?;
    }
    if actual != package.entries { return Err(invalid("extracted source differs from locked archive")); }
    Ok(root)
}
fn install(project: &Path, locked: &Locked, bytes: &[u8], package: &Package) -> Result<()> {
    let parent = cache_parent(project, true)?;
    let destination = parent.join(&locked.sha256);
    if destination.try_exists().map_err(io)? { verify_cache(project, locked)?; return Ok(()); }
    let stage = parent.join(format!(".stage-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    fs::create_dir(&stage).map_err(io)?;
    let result = (|| {
        publish(&stage.join("archive.tar.gz"), bytes)?;
        let root = stage.join(package.manifest.root());
        fs::create_dir(&root).map_err(io)?;
        for (path, contents) in &package.entries {
            let path = root.join(path);
            if let Some(bytes) = contents { publish(&path, bytes)?; }
            else { fs::create_dir(&path).map_err(io)?; }
        }
        #[cfg(unix)] {
            for (path, contents) in package.entries.iter().rev() {
                if contents.is_none() { fs::File::open(root.join(path)).map_err(io)?.sync_all().map_err(io)?; }
            }
            fs::File::open(&root).map_err(io)?.sync_all().map_err(io)?;
            fs::File::open(&stage).map_err(io)?.sync_all().map_err(io)?;
        }
        fs::rename(&stage, &destination).map_err(io)?;
        #[cfg(unix)] fs::File::open(&parent).map_err(io)?.sync_all().map_err(io)?;
        Ok(())
    })();
    if stage.exists() { let _ = fs::remove_dir_all(&stage); }
    result
}

fn download(url: &str, limit: usize, local_fixture: bool) -> Result<Vec<u8>> {
    let mut command = Command::new("curl");
    command.args(["--disable", "--globoff", "--silent", "--show-error", "--fail", "--location",
        "--max-redirs", "5", "--connect-timeout", "10", "--max-time", "60", "--proto",
        if local_fixture { "=http" } else { "=https" }, "--proto-redir",
        if local_fixture { "=http" } else { "=https" }, "--max-filesize", &limit.to_string(), url])
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = command.spawn().map_err(io)?;
    let mut bytes = Vec::new();
    let read = child.stdout.take().unwrap().take(limit as u64 + 1).read_to_end(&mut bytes);
    if read.is_err() || bytes.len() > limit { let _ = child.kill(); }
    let status = child.wait().map_err(io)?;
    read.map_err(io)?;
    if bytes.len() > limit { return Err("LANA_ERR_LIMIT".into()); }
    if !status.success() { return Err(format!("LANA_ERR_IO: package download failed ({status})")); }
    Ok(bytes)
}
fn release_origin() -> Result<(String, bool)> {
    #[cfg(feature = "package-test-origin")]
    if let Ok(origin) = std::env::var("LANA_TEST_PACKAGE_ORIGIN") {
        let port = origin.strip_prefix("http://127.0.0.1:").ok_or_else(|| invalid("fixture must be loopback HTTP"))?;
        if port.parse::<u16>().is_err() { return Err(invalid("fixture port")); }
        return Ok((origin, true));
    }
    Ok(("https://github.com".into(), false))
}
fn fetch_closure(project: &Path, key: &str, old: &Lock, fetched: &mut BTreeMap<String, Locked>,
    visiting: &mut BTreeSet<String>, origin: &str, local: bool) -> Result<()> {
    let (id, ver) = exact(key)?;
    if visiting.contains(id) { return Err(invalid("package dependency cycle")); }
    if let Some(existing) = fetched.get(id) {
        return if existing.version == ver { Ok(()) } else { Err(invalid("conflicting package versions")) };
    }
    if fetched.len() + visiting.len() >= 64 { return Err("LANA_ERR_LIMIT".into()); }
    if old.packages.iter().any(|entry| entry.identity == id && entry.version != ver) { return Err(invalid("locked package version conflict")); }
    visiting.insert(id.into());
    let repo = id.split('/').nth(1).unwrap();
    let tag = format!("lana-v{ver}"); let asset = format!("{repo}-{ver}-lana.tar.gz");
    let base = format!("{origin}/{id}/releases/download/{tag}");
    let sums = download(&format!("{base}/SHA256SUMS"), 1024, local)?;
    let sums = std::str::from_utf8(&sums).map_err(|_| invalid("checksum UTF-8"))?;
    let digest = sums.get(..64).ok_or_else(|| invalid("checksum line"))?;
    if !digest.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) || sums != format!("{digest}  {asset}\n") {
        return Err(invalid("exact checksum entry required"));
    }
    if old.packages.iter().any(|entry| entry.identity == id && entry.sha256 != digest) { return Err(invalid("release asset changed from locked digest")); }
    let bytes = download(&format!("{base}/{asset}"), COMPRESSED, local)?;
    if sha(&bytes) != digest { return Err(invalid("download digest mismatch")); }
    let package = unpack(&bytes)?;
    if package.manifest.name != repo || package.manifest.version != ver { return Err(invalid("release manifest mismatch")); }
    let locked = Locked { identity:id.into(), version:ver.into(), tag, asset, sha256:digest.into(), direct:false,
        dependencies:package.manifest.dependencies() };
    install(project, &locked, &bytes, &package)?;
    drop(package);
    drop(bytes);
    for dependency in &locked.dependencies { fetch_closure(project, dependency, old, fetched, visiting, origin, local)?; }
    visiting.remove(id); fetched.insert(id.into(), locked); Ok(())
}
fn verified(project: &Path) -> Result<Option<(Lock, HashMap<String, String>)>> {
    let Some(lock) = load_lock(project)? else { return Ok(None); };
    let mut paths = HashMap::new();
    for entry in &lock.packages {
        let path = verify_cache(project, entry)?;
        paths.insert(entry.identity.clone(), fs::canonicalize(path).map_err(io)?.to_string_lossy().into_owned());
    }
    Ok(Some((lock, paths)))
}
pub fn compiler_paths(input: &Path) -> Result<HashMap<String, String>> {
    let input = fs::canonicalize(input).map_err(io)?;
    let start = if input.is_dir() { input.as_path() } else { input.parent().unwrap() };
    for directory in start.ancestors() {
        if directory.join("lana.toml").is_file() {
            return Ok(verified(directory)?.map(|(_, paths)| paths).unwrap_or_default());
        }
    }
    Ok(HashMap::new())
}
pub fn build_hash(project: &Path, mut hash: u64) -> Result<(u64, bool)> {
    let Some((lock, _)) = verified(project)? else { return Ok((hash, false)); };
    for byte in canonical(&lock)? { hash ^= u64::from(byte); hash = hash.wrapping_mul(0x100000001b3); }
    Ok((hash, true))
}
fn add(project: &Path, key: &str) -> Result<Value> {
    exact(key)?;
    if !project.join("lana.toml").is_file() { return Err(invalid("lana.toml not found")); }
    let old = verified(project)?.map(|(lock, _)| lock).unwrap_or_else(Lock::empty);
    let mut direct = old.direct.clone(); direct.push(key.into()); direct.sort(); direct.dedup();
    let (origin, local) = release_origin()?;
    let mut fetched = BTreeMap::new();
    for key in &direct { fetch_closure(project, key, &old, &mut fetched, &mut BTreeSet::new(), &origin, local)?; }
    let mut packages = fetched.into_values().collect::<Vec<_>>();
    for package in &mut packages { package.direct = direct.contains(&package.exact()); }
    let lock = Lock { schema_version:1, direct, packages };
    lock.validate()?;
    if lock != old { publish(&project.join("lana.lock"), &canonical(&lock)?)?; }
    Ok(json!({"status":"ok","changed":lock != old,"packages":lock.packages.len(),"direct":lock.direct}))
}
pub fn command(args: &[String]) -> Result<Value> {
    match args.first().map(String::as_str) {
        Some("pack") if args.len() == 4 && args[2] == "-o" => {
            let package = from_directory(Path::new(&args[1]))?;
            let bytes = archive(&package)?;
            unpack(&bytes)?;
            let output = Path::new(&args[3]);
            let source = fs::canonicalize(&args[1]).map_err(io)?;
            let parent = fs::canonicalize(output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."))).map_err(io)?;
            if parent.starts_with(source.join("src")) || parent.starts_with(source.join("tests")) || parent.join(output.file_name().unwrap_or_default()) == source.join("lana.toml") {
                return Err(invalid("archive output overlaps source tree"));
            }
            publish(output, &bytes)?;
            Ok(json!({"status":"ok","archive":output,"sha256":sha(&bytes),"name":package.manifest.name,
                "version":package.manifest.version,"asset":format!("{}-lana.tar.gz",package.manifest.root())}))
        }
        Some("add") if args.len() == 2 => add(&std::env::current_dir().map_err(io)?, &args[1]),
        _ => Err("usage: lana package pack DIRECTORY -o ARCHIVE | lana package add owner/repo@X.Y.Z".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture_package() -> Package {
        let manifest = b"schema = 1\nname = \"demo\"\nversion = \"1.2.3\"\nentry = \"src/main.lana\"\n";
        Package { manifest: Manifest::parse(manifest).unwrap(), entries: BTreeMap::from([
            ("lana.toml".into(), Some(manifest.to_vec())), ("src".into(), None),
            ("src/main.lana".into(), Some(b"fn value() { return 7; }\n".to_vec()))]) }
    }
    #[test]
    fn deterministic_archive_and_strict_reader() {
        let package = fixture_package(); let bytes = archive(&package).unwrap();
        assert_eq!(bytes, archive(&package).unwrap());
        assert_eq!(&bytes[3..8], &[0, 0, 0, 0, 0]); assert_eq!(bytes[9], 255);
        assert_eq!(unpack(&bytes).unwrap().entries, package.entries);
        assert_eq!(sha(&bytes), "67d67ace954fafd6ff53563a0611a38ef4abae2ba5edde1ce4b1febc8806489d");
        assert!(unpack(&bytes[..bytes.len()-1]).is_err());
        let mut trailing = bytes.clone(); trailing.push(0); assert!(unpack(&trailing).is_err());
        let mut bad = fixture_package(); bad.entries.insert("src/../escape".into(), Some(vec![]));
        assert!(archive(&bad).is_err());
        let mut bad = fixture_package(); bad.entries.insert("src/MAIN.lana".into(), Some(vec![]));
        assert!(unpack(&archive(&bad).unwrap()).is_err());
        let mut bad = fixture_package(); bad.entries.insert("outside".into(), Some(vec![]));
        assert!(unpack(&archive(&bad).unwrap()).is_err());
        for id in ["Owner/repo@1.0.0", "owner/repo@01.0.0", "owner/repo@1.0", "../repo@1.0.0", "owner/repo@1.0.0-beta"] {
            assert!(exact(id).is_err());
        }
    }
    #[test]
    fn locked_cache_rejects_changed_source_and_closure() {
        let root = std::env::temp_dir().join(format!("lana-package-unit-{}",std::process::id()));
        fs::create_dir(&root).unwrap();
        let package = fixture_package(); let bytes = archive(&package).unwrap();
        let entry = Locked { identity:"owner/demo".into(),version:"1.2.3".into(), tag:"lana-v1.2.3".into(),
            asset:"demo-1.2.3-lana.tar.gz".into(),sha256:sha(&bytes),direct:true,dependencies:vec![] };
        let lock = Lock { schema_version:1,direct:vec![entry.exact()],packages:vec![entry.clone()] };
        lock.validate().unwrap(); install(&root, &entry, &bytes, &package).unwrap();
        let source = verify_cache(&root, &entry).unwrap().join("src/main.lana");
        fs::write(&source, "changed").unwrap(); assert!(verify_cache(&root, &entry).is_err());
        fs::write(&source, package.entries["src/main.lana"].as_ref().unwrap()).unwrap();
        verify_cache(&root, &entry).unwrap();
        let mut cycle = lock.clone(); cycle.packages[0].dependencies.push(entry.exact()); assert!(cycle.validate().is_err());
        let mut missing = lock.clone(); missing.packages[0].dependencies.push("owner/other@1.0.0".into()); assert!(missing.validate().is_err());
        fs::remove_dir_all(&root).unwrap();
    }
    #[test]
    #[ignore = "explicit public GitHub download and checksum smoke test"]
    fn public_release_smoke() {
        let base = "https://github.com/cli/cli/releases/download/v2.63.2";
        let sums = download(&format!("{base}/gh_2.63.2_checksums.txt"), 64*1024, false).unwrap();
        let bytes = download(&format!("{base}/gh_2.63.2_linux_amd64.tar.gz"), COMPRESSED, false).unwrap();
        let line = format!("{}  gh_2.63.2_linux_amd64.tar.gz", sha(&bytes));
        assert!(std::str::from_utf8(&sums).unwrap().lines().any(|entry| entry == line));
        eprintln!("PUBLIC_RELEASE_DIGEST_PASS {} bytes {}", bytes.len(), sha(&bytes));
    }
}
