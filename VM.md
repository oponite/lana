# C Virtual Machine

The canonical VM is a C11 register machine. Each frame has 256 tagged-value
registers and register-owned history. Semantic functions live outside the
opcode dispatcher.

Runtime heap objects use VM-lifetime allocation and are released by `ss_vm_free`.
The VM reports allocation failure instead of exiting. Seeded sample behavior
uses PCG32 for cross-platform repeatability.

The public Python `ss` command compiles source and invokes the native `ssvm`
binary. `ssvm` can also assemble, verify, disassemble, trace, and execute SSBC
directly without Python.

