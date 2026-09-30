// Shared Cargo build script for the CLI and WASM compiler artifact.
use std::{env, fs, path::PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source = manifest.join("../../../compiler/bootstrap/compiler.lasm");
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("lana-compiler.labc");
    let assembly = fs::read_to_string(&source).expect("read checked compiler bootstrap");
    let chunk = lana_bytecode::assemble(&assembly).expect("assemble checked compiler bootstrap");
    lana_bytecode::verifier::verify(&chunk).expect("verify compiler bootstrap");
    fs::write(&output, lana_bytecode::encoder::encode(&chunk)).expect("write compiler artifact");
    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rustc-env=LANA_BUILT_COMPILER={}", output.display());
    if env::var("CARGO_PKG_NAME").unwrap() == "lana-wasm" {
        let stdlib = manifest.join("../../../stdlib");
        println!("cargo:rerun-if-changed={}", stdlib.display());
        let mut paths: Vec<_> = fs::read_dir(&stdlib).expect("read stdlib")
            .map(|entry| entry.expect("stdlib entry").path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "lana")).collect();
        paths.sort();
        let mut embedded = String::from("const STDLIB: &[(&str, &str)] = &[\n");
        for path in paths {
            let name = format!("stdlib/{}", path.file_name().unwrap().to_str().unwrap());
            let text = fs::read_to_string(&path).expect("read stdlib module");
            embedded.push_str(&format!("({name:?}, {text:?}),\n"));
        }
        embedded.push_str("];\n");
        fs::write(output.with_file_name("stdlib.rs"), embedded).expect("embed stdlib");
    }
}
