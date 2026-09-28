//! Atomic replacement shared by brain packages and compiler output.
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct UncertainWrite;

impl std::fmt::Display for UncertainWrite {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("file replaced; durability uncertain; inspect destination before retry")
    }
}

impl std::error::Error for UncertainWrite {}

pub fn durability_uncertain(error: &io::Error) -> bool {
    error.get_ref().is_some_and(|cause| cause.is::<UncertainWrite>())
}

/// Validation/write/rename errors leave the old destination intact. A directory
/// sync error after rename means the complete new file is visible but its crash
/// durability is uncertain; callers must inspect it before retrying.
pub fn write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    write_with_sync(path, bytes,
        |file| { publication_failpoint("before_file_sync")?; file.sync_all() },
        || publication_failpoint("before_rename"),
        |directory| { publication_failpoint("after_rename")?; directory.sync_all() })
}

// Opt-in test binary only; ordinary builds cannot read this failpoint.
fn publication_failpoint(stage: &str) -> io::Result<()> {
    #[cfg(feature = "publication-fault-injection")]
    if std::env::var("LANA_TEST_ATOMIC_STAGE").as_deref() == Ok(stage) {
        return Err(io::Error::other("injected publication failure"));
    }
    let _ = stage;
    Ok(())
}

fn write_with_sync(
    path: &Path,
    bytes: &[u8],
    sync_file: impl FnOnce(&fs::File) -> io::Result<()>,
    before_rename: impl FnOnce() -> io::Result<()>,
    sync_parent: impl FnOnce(&fs::File) -> io::Result<()>,
) -> io::Result<()> {
    #[cfg(not(unix))]
    let _ = sync_parent;
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
        sync_file(&file)?;
        drop(file);
        before_rename()?;
        fs::rename(&temporary, path)?;
        #[cfg(unix)]
        sync_parent(&directory).map_err(|_| io::Error::other(UncertainWrite))?;
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

#[cfg(unix)]
#[test]
fn pre_and_post_replacement_errors_have_distinct_outcomes() {
    let root = std::env::temp_dir().join(format!("lana-atomic-outcomes-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("value");
    write(&path, b"old").unwrap();
    let before = write_with_sync(&path, b"new", |_| Err(io::Error::other("before")), || Ok(()), |_| Ok(())).unwrap_err();
    assert!(!durability_uncertain(&before));
    assert_eq!(fs::read(&path).unwrap(), b"old");
    let after = write_with_sync(&path, b"new", |_| Ok(()), || Ok(()), |_| Err(io::Error::other("after"))).unwrap_err();
    assert!(durability_uncertain(&after));
    assert_eq!(fs::read(&path).unwrap(), b"new");
    assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
    fs::remove_dir_all(root).unwrap();
}

// Failpoints exist only in the test executable, never in normal builds.
#[cfg(unix)]
#[test]
fn recovery_process() {
    let Some(root) = std::env::var_os("LANA_ATOMIC_TEST_ROOT") else { return; };
    let root = std::path::PathBuf::from(root);
    let action = std::env::var("LANA_ATOMIC_TEST_ACTION").unwrap();
    let destination = root.join("destination");
    if action == "inspect" {
        let expected = root.join(std::env::var("LANA_ATOMIC_TEST_EXPECTED").unwrap());
        if !expected.exists() { assert!(!destination.exists()); return; }
        assert_eq!(fs::read(&destination).unwrap(), fs::read(&expected).unwrap());
        assert_eq!(crate::brain::Brain::load(&destination).unwrap(), crate::brain::Brain::load(&expected).unwrap());
        return;
    }
    let stage: u8 = std::env::var("LANA_ATOMIC_TEST_STAGE").unwrap().parse().unwrap();
    let fault = |current| {
        if stage != current { return Ok(()); }
        if action == "exit" { std::process::exit(73); }
        Err(io::Error::other("injected write failure"))
    };
    let bytes = fs::read(root.join("new")).unwrap();
    let error = write_with_sync(&destination, &bytes,
        |file| { fault(0)?; file.sync_all() },
        || fault(1),
        |directory| { fault(2)?; directory.sync_all() }).unwrap_err();
    assert_eq!(durability_uncertain(&error), stage == 2);
}

#[cfg(unix)]
#[test]
fn brain_recovery_after_failures_and_process_exit() {
    use crate::brain::{Activation, Brain};
    let root = std::env::temp_dir().join(format!("lana-atomic-recovery-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    for format in 0..3 {
        let mut old = if format != 0 {
            Brain::new_layers(3, 2, &[(4, Activation::Relu), (2, Activation::Gelu)], 7).unwrap()
        } else { Brain::new(3, 2, 4, 7).unwrap() };
        if format == 2 {
            let mut memory = crate::brain_memory::Memory::empty();
            memory.add("known", "recovery fixture", crate::information_codec::Tagged::Definite {
                value: Box::new(crate::information_codec::Tagged::Bool { value: true }),
            }).unwrap();
            old.typed_memory_json = crate::information_codec::canonical(&memory).unwrap();
        }
        let mut new = old.clone();
        new.train_next_token(&[0, 1], 2, 0.1).unwrap();
        old.save(&root.join("old")).unwrap();
        new.save(&root.join("new")).unwrap();
        for present in [false, true] {
            for stage in 0..3 {
                for action in ["error", "exit"] {
                    let destination = root.join("destination");
                    if destination.exists() { fs::remove_file(&destination).unwrap(); }
                    if present { fs::copy(root.join("old"), &destination).unwrap(); }
                    let child = |action: &str| {
                        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
                        command.args(["--exact", "atomic_file::recovery_process", "--nocapture"])
                            .env("LANA_ATOMIC_TEST_ROOT", &root)
                            .env("LANA_ATOMIC_TEST_ACTION", action)
                            .env("LANA_ATOMIC_TEST_STAGE", stage.to_string());
                        command
                    };
                    let result = child(action).output().unwrap();
                    assert_eq!(result.status.code(), Some(if action == "exit" { 73 } else { 0 }),
                        "{}{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
                    let expected = if stage == 2 { "new" } else if present { "old" } else { "absent" };
                    let result = child("inspect").env("LANA_ATOMIC_TEST_EXPECTED", expected).output().unwrap();
                    assert!(result.status.success(), "{}{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
                }
            }
        }
    }
    fs::remove_dir_all(root).unwrap();
}
