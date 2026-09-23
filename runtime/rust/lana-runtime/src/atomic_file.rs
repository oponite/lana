//! Atomic replacement shared by brain packages and compiler output.
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

/// Validation/write/rename errors leave the old destination intact. A directory
/// sync error after rename means the complete new file is visible but its crash
/// durability is uncertain; callers must inspect it before retrying.
pub fn write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let name = path.file_name().ok_or_else(|| io::Error::other("missing file name"))?;
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    #[cfg(unix)]
    let directory = fs::File::open(parent)?;
    let mut name = name.to_os_string();
    name.push(format!(".lana-{}-{}.tmp", std::process::id(), NEXT_FILE.fetch_add(1, Ordering::Relaxed)));
    let temporary = parent.join(name);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)] {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary)?;
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        #[cfg(unix)]
        directory.sync_all().map_err(|_| io::Error::other("file replaced; durability uncertain; inspect destination before retry"))?;
        Ok(())
    })();
    if result.is_err() { let _ = fs::remove_file(&temporary); }
    result
}

#[test]
fn replacement_failure_preserves_destination_and_cleans_owned_file() {
    let root = std::env::temp_dir().join(format!("lana-atomic-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("value");
    write(&path, b"old").unwrap();
    write(&path, b"new").unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"new");
    let directory = root.join("directory");
    fs::create_dir_all(&directory).unwrap();
    assert!(write(&directory, b"invalid").is_err());
    assert!(directory.is_dir());
    assert_eq!(fs::read_dir(&root).unwrap().count(), 2);
    fs::remove_dir_all(root).unwrap();
}
