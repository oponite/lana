# Lana 1.0 Launch Roadmap

This file lists unfinished launch work only.

## 1. Data-complete programs

- Add maps and local module imports.
- Add native CSV reading and JSON parsing/stringifying.
- Define consistent file, parse, assertion, and task error messages.

## 2. Concurrency hardening

- Replace one-pthread-per-task execution with a bounded worker pool.
- Preserve configured state history when a state is passed to a task.
- Add ThreadSanitizer coverage, scheduler stress tests, and malformed task-bytecode fuzzing.

## 3. Out-of-the-box distribution

- Publish a signed macOS ARM64 wheel and Homebrew bottle.
- Test installation and execution on a clean Apple Silicon runner.
- Make `lana test` source locations and assertion failures release-quality.

## 4. Launch evidence

- Add CSV and concurrent evidence examples to the five-minute guide.
- Rerun the falsification benchmark on the release build.
- Freeze the Lana 1.0 source specification after release-candidate tests pass.
