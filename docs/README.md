# Documentation

This documentation describes the behavior in the current repository source.
For individual items and signatures, use the
[API documentation](https://docs.rs/gguf-rs-lib).

## Guides

- [Format and type support](format-support.md) — accepted GGUF versions,
  metadata types, tensor identifiers, byte order, and payload limitations.
- [Feature semantics](features.md) — `std`, `alloc`, `async`, `mmap`, and
  supported build combinations.
- [Safety and validation](safety-and-validation.md) — parser limits, checks
  performed, checks not performed, and the memory-map boundary.
- [CLI guide](cli.md) — source installation, implemented commands, output
  formats, validation, and comparison behavior.
- [Examples](examples.md) — runnable example catalog and commands.
- [Changelog](../CHANGELOG.md) — user-facing changes and release history.
- [Security policy](../SECURITY.md) — supported versions and private reporting.
- [Testing guide](../TESTING_GUIDE.md) — local verification and coverage.
- [Contributing guide](../CONTRIBUTING.md) — contributor workflow and pull
  request expectations.

## Choosing an API

| Goal | API |
| --- | --- |
| Read metadata and tensor descriptors from a seekable source | `GGUFFileReader` |
| Read from a non-seekable `Read` stream | `GGUFStreamReader` |
| Create a GGUF file from unquantized raw data | `GGUFBuilder` |
| Control low-level writing order | `GGUFFileWriter` or `GGUFStreamWriter` |
| Use data structures without `std` | Disable defaults and enable `alloc` |
| Validate a magic/version prefix asynchronously | `AsyncGGUFFile` (preview) |
| Map a file and validate its magic/version prefix | `MmapGGUFFile` (preview) |

The synchronous reader is the recommended inspection path. Optional async and
memory-map types do not yet parse metadata or tensor descriptors.

## Project identifiers

- Repository: [ThreatFlux/gguf](https://github.com/ThreatFlux/gguf)
- Published library crate:
  [`gguf-rs-lib`](https://crates.io/crates/gguf-rs-lib)
- Rust import path: `gguf_rs_lib`
- Workspace CLI binary: `gguf-cli`

The crates.io package named `gguf` is not this project.
