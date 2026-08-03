# gguf-rs

[![Crates.io](https://img.shields.io/crates/v/gguf-rs-lib.svg)](https://crates.io/crates/gguf-rs-lib)
[![Documentation](https://docs.rs/gguf-rs-lib/badge.svg)](https://docs.rs/gguf-rs-lib)
[![CI](https://github.com/ThreatFlux/gguf/actions/workflows/ci.yml/badge.svg)](https://github.com/ThreatFlux/gguf/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`gguf-rs-lib` is a Rust library for inspecting and creating
[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) files. Its
stable path is synchronous GGUF v3 header, metadata, and tensor-descriptor I/O.
The API also includes tensor data helpers and optional preview APIs for Tokio
and memory mapping.

The crate does not execute models, quantize tensors, or provide inference.

## Support at a glance

| Capability | Status |
| --- | --- |
| GGUF version | Version 3 only |
| Byte order | Little-endian only |
| Metadata | All GGUF value IDs 0–12; validated ASCII hierarchical keys |
| Tensor descriptors | Canonical type IDs 0–3, 6–30, 34–35, and 39–42 |
| Tensor dimensions | Exactly 1–4 dimensions per descriptor |
| Unquantized payloads | Raw read/write with exact size checks |
| Quantized payloads | Exact recognized block sizes; no codec |
| Synchronous reader/writer | Implemented with `std` |
| `no_std` | Data structures and helpers with `alloc`; no file/stream I/O |
| Tokio `async` | Preview: validates only magic and version |
| `mmap` | Preview: maps a file and validates only magic and version |
| CLI | Workspace inspection, validation, and comparison tool |

See [format support](docs/format-support.md) and
[features](docs/features.md) before using the crate with quantized model
payloads, async I/O, memory mapping, or untrusted files.

## Install

The library package on crates.io is `gguf-rs-lib`:

```toml
[dependencies]
gguf-rs-lib = "0.3"
```

The similarly named crates.io package `gguf` is unrelated. The source tree can
be ahead of the latest published `gguf-rs-lib` release; pin a Git revision when
depending directly on the repository. Version 0.3 corrects public tensor-type
discriminants; applications that cast or persist `GGUFTensorType` values should
review the [migration note](CHANGELOG.md#migration-from-02x).

## Read a file

```rust
use gguf_rs_lib::reader::GGUFFileReader;
use gguf_rs_lib::Result;
use std::fs::File;

fn main() -> Result<()> {
    let reader = GGUFFileReader::new(File::open("model.gguf")?)?;

    println!("GGUF v{}", reader.header().version);
    println!("{} metadata entries", reader.metadata().len());
    println!("{} tensor descriptors", reader.tensor_count());

    if let Some(name) = reader.metadata().get_string("general.name") {
        println!("model: {name}");
    }

    for tensor in reader.tensor_infos().iter().take(10) {
        println!(
            "{}: {} {:?}",
            tensor.name(),
            tensor.tensor_type().name(),
            tensor.shape().dims()
        );
    }

    Ok(())
}
```

`GGUFFileReader::new` parses the header, metadata, and tensor descriptors and
validates that declared payload ranges fit the source. It does not load tensor
payload bytes by default. Use `load_tensor_data` or `load_all_tensor_data` when
those bytes are needed.

Run the complete example with:

```bash
cargo run --locked --example basic_usage -- model.gguf
```

## Create a file

```rust
use gguf_rs_lib::builder::GGUFBuilder;
use gguf_rs_lib::Result;

fn main() -> Result<()> {
    let result = GGUFBuilder::simple("tiny-model", "GGUF writer example")
        .add_f32_tensor("weights", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?
        .build_to_file("tiny-model.gguf")?;

    println!("wrote {} bytes", result.total_bytes_written);
    Ok(())
}
```

The high-level builder is the recommended writing API. It computes relative
tensor offsets and validates raw payload lengths for the represented tensor
type. Tensor convenience methods return `Result<GGUFBuilder>` so invalid ranks,
payload sizes, and quantization block geometry fail at the call site.
Quantized inputs must already be encoded in the recognized GGML block layout;
the crate does not quantize values. The builder adds and validates
`general.quantization_version = 2` whenever a quantized tensor is present.

## Features

| Feature | Default | What it enables |
| --- | --- | --- |
| `std` | Yes | File/stream I/O, builders, errors, Serde derives |
| `alloc` | Via `std` | Allocating data structures for `no_std` builds |
| `async` | No | Tokio-based, header-only preview reader; implies `std` |
| `mmap` | No | Memory-map and byte-reader preview APIs; implies `std` |

The minimum supported Rust version is 1.87.

Useful checks:

```bash
cargo check --locked -p gguf-rs-lib --no-default-features --features alloc
cargo check --locked -p gguf-rs-lib --all-features
```

A bare `--no-default-features` build is not a supported configuration; enable
`alloc`. Details and examples are in [the feature guide](docs/features.md).

## CLI

`gguf-cli` is currently a workspace-only package. Install it from a clone:

```bash
git clone https://github.com/ThreatFlux/gguf.git
cd gguf
cargo install --locked --path gguf-cli
```

Implemented paths include:

```bash
gguf-cli info model.gguf --detailed
gguf-cli tensors model.gguf
gguf-cli tensors model.gguf --summary
gguf-cli metadata model.gguf --format json
gguf-cli metadata model.gguf --format yaml --key general.
gguf-cli validate models/ --recursive --integrity
gguf-cli compare before.gguf after.gguf
gguf-cli compare before.gguf after.gguf --data
```

The CLI uses the complete synchronous parser. Directory validation checks
direct `.gguf` children or all descendants with `--recursive`; `--integrity`
also reads every declared tensor payload. Comparison checks metadata and tensor
descriptors by default and exact payload bytes with `--data`. These operations
do not provide a cryptographic integrity, authenticity, or model-correctness
guarantee. See the [CLI guide](docs/cli.md).

## Performance and safety

No cross-language or absolute performance claim is made. The synchronous
reader parses metadata and descriptors eagerly and tensor bytes on request.
Benchmark the operations, files, and feature set that match your workload.

The default feature set does not execute an `unsafe` block. The `mmap` feature
has an explicitly unsafe constructor because callers must ensure a mapped file
is not concurrently truncated or modified. Parsing also has defensive limits,
but it is not a complete semantic, checksum, or authenticity validation of a
model.

Read [safety and validation](docs/safety-and-validation.md) for the exact
boundaries.

## Documentation

- [Documentation index](docs/README.md)
- [Changelog](CHANGELOG.md)
- [Format and type support](docs/format-support.md)
- [Feature semantics](docs/features.md)
- [Safety and validation](docs/safety-and-validation.md)
- [CLI guide](docs/cli.md)
- [Examples](docs/examples.md)
- [Testing guide](TESTING_GUIDE.md)
- [Contributing](CONTRIBUTING.md)

API documentation is published on [docs.rs](https://docs.rs/gguf-rs-lib).

## Development

```bash
git clone https://github.com/ThreatFlux/gguf.git
cd gguf

cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
python3 scripts/check_docs.py
./scripts/check_package.sh
```

The workspace tracks `Cargo.lock` because it contains the `gguf-cli`
application. Use `--locked` in CI and release-oriented commands.

## Security

See the [security policy](SECURITY.md). Report suspected vulnerabilities
through
[GitHub private vulnerability reporting](https://github.com/ThreatFlux/gguf/security/advisories/new);
do not open a public issue for an undisclosed vulnerability.

## License

Licensed under the [MIT License](LICENSE).
