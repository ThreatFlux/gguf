# Examples

Examples are built from the repository root. Use `--locked` so the workspace
application and development dependencies resolve exactly as reviewed.

| Example | Purpose | Command |
| --- | --- | --- |
| `basic_usage` | Parse metadata and tensor descriptors | `cargo run --locked --example basic_usage -- model.gguf` |
| `inspect_gguf` | Print a more detailed, sorted inspection | `cargo run --locked --example inspect_gguf -- model.gguf` |
| `create_test_gguf` | Create and read back a small unquantized file | `cargo run --locked --example create_test_gguf` |
| `roundtrip_test` | Exercise an in-memory builder/reader round trip | `cargo run --locked --example roundtrip_test` |
| `async_usage` | Demonstrate async prefix validation | `cargo run --locked --example async_usage --features async -- model.gguf` |

`create_test_gguf` writes to `target/examples/test-model.gguf` by default so a
generated binary does not appear in the repository root. Pass an explicit
output path after `--` to keep the file elsewhere.

The async example deliberately reports only the version. The preview async API
does not parse metadata or tensors.

## Build without running

```bash
cargo build --locked -p gguf-rs-lib --examples --all-features
```

## Using real model files

This repository does not check in full model files. Pass your own file to an
inspection example and keep large models under a local ignored directory such
as `data/`.

Before loading tensor payloads from a real quantized model, read
[format and type support](format-support.md). The crate calculates exact raw
block sizes for recognized types but does not dequantize or validate numeric
model semantics.

## Expected failures

An example should return a nonzero status for an invalid path, unsupported
version, malformed metadata, unknown tensor type, or other parse error. The
examples do not silently classify such a file as compatible.
