# Security

## Trust model

Lana runs programs with the full authority of the user who starts it.

- **Programs are trusted.** Lana does not support running untrusted source or
  bytecode. Resource limits (heap, instructions, bytecode size) guard against
  mistakes and runaway programs; they are not a sandbox.
- **Integration clients are trusted.** The Python worker, Jupyter extension,
  and MCP server run programs that their client supplies. The MCP server's
  `--allow-run` lets the connected client run any program under `--root`;
  enable it only for clients you would trust with a shell.
- **Store directories must be owned by the user.** Do not place stores in
  shared or other-user-writable directories. Journal checks that reject
  symlinks and shared hard links are defense in depth, not a supported
  boundary.

Reports that depend on running an untrusted program, or on another user
writing to a store directory, are outside this model.

## Reporting

Report vulnerabilities privately through GitHub:
**Security → Report a vulnerability** on
[oponite/lana](https://github.com/oponite/lana/security/advisories/new).
Do not open a public issue. Include the Lana version (`lana version`), the
platform, and a minimal reproduction.
