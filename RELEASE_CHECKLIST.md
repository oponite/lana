# Lana 1.0 release checklist

- [ ] `ctest --test-dir build --output-on-failure` passes, including native bootstrap and import tests.
- [ ] Version consistency check reports `1.0.0` for `VERSION`, CMake, and the native CLI.
- [ ] Universal `ssvm` is signed and `lipo -archs` reports `arm64 x86_64`.
- [ ] The same native archive runs on macOS 13+ Intel and Apple Silicon.
- [ ] ASan/UBSan, ThreadSanitizer, JSON/CSV/bytecode fuzz, and Lana tests pass.
- [ ] Native archive checksums and attestations are recorded.
- [ ] Homebrew Intel and ARM bottles install and pass `lana version`.
- [ ] SPEC.md, BYTECODE.md, and VM.md are frozen after all gates pass.
