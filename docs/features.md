# Feature semantics

The default `std` build is the primary supported configuration. Feature names
describe dependency and API availability; they do not by themselves imply
that every high-level operation has a specialized implementation.

The minimum supported Rust version (MSRV) is 1.87, as declared by both
workspace packages.

## Feature matrix

| Feature | Implies | Available behavior |
| --- | --- | --- |
| `std` | `alloc`, Serde and `thiserror` support | File/stream I/O, high-level builder, standard errors |
| `alloc` | `hashbrown` and `libm` | Allocating metadata/tensor data structures in `no_std` |
| `async` | `std` and Tokio | Async magic/version reads only |
| `mmap` | `std` and `memmap2` | Mapping, prefix validation, and primitive byte reads |

## Default synchronous build

Use the default feature set for complete header, metadata, descriptor, and
raw-payload I/O:

```toml
[dependencies]
gguf-rs-lib = "0.3"
```

The synchronous file reader requires `Read + Seek`. For non-seekable input, use
`GGUFStreamReader<R: Read>`.

## `no_std` with allocation

Disable default features and explicitly enable `alloc`:

```toml
[dependencies]
gguf-rs-lib = { version = "0.3", default-features = false, features = ["alloc"] }
```

Verify this configuration with:

```bash
cargo check --locked -p gguf-rs-lib --no-default-features --features alloc
```

This configuration provides metadata, tensor, format, and helper data
structures that can use an allocator. File readers, stream readers, file
writers, stream writers, and `GGUFBuilder` are gated on `std`.

A bare `--no-default-features` build is not supported. The public data model
contains strings, vectors, maps, and boxed arrays and therefore requires
allocation for meaningful use.

## Async preview

Enable Tokio support:

```toml
[dependencies]
gguf-rs-lib = { version = "0.3", features = ["async"] }
```

`AsyncGGUFFile::read_async` and `read_file_async` asynchronously read and
validate only the four-byte magic and four-byte version. The returned metadata
and tensor collections are currently empty even when the source contains them.
Do not use this type to count, validate, or access a file's metadata or
tensors.

```rust
use gguf_rs_lib::r#async::AsyncGGUFFile;
use gguf_rs_lib::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let header = AsyncGGUFFile::read_file_async("model.gguf").await?;
    println!("validated GGUF v{} prefix", header.version);
    Ok(())
}
```

For a complete parse in an async application today, isolate the synchronous
reader on a blocking thread and account for the file I/O and allocation costs.

## Memory-map preview

Enable memory mapping:

```toml
[dependencies]
gguf-rs-lib = { version = "0.3", features = ["mmap"] }
```

The unsafe `MmapGGUFFile::mmap` constructor maps a file and validates its magic
and version. It does not parse metadata or tensor descriptors. The validated
view exposes its version, length, complete byte slice, and a bounded
`MmapGGUFReader`.
`MmapGGUFReader` provides bounded seek, byte-slice, `u32`, and `u64` reads over
an existing `Arc<Mmap>`.

`TensorData::new_mapped` can expose a bounded region without copying once the
caller has independently established a correct payload offset and length.
The high-level synchronous file reader does not switch to memory mapping when
`GGUFReaderConfig::use_mmap` is set.

Creating an OS memory map requires an unsafe call because file stability cannot
be enforced by Rust's type system. Keep mapped files immutable for the lifetime
of the map; see
[safety and validation](safety-and-validation.md#memory-mapping).

## Build all supported configurations

```bash
cargo check --locked -p gguf-rs-lib
cargo check --locked -p gguf-rs-lib --all-features
cargo check --locked -p gguf-rs-lib --no-default-features --features alloc
cargo check --locked -p gguf-cli --all-features
```

The CLI is a workspace application and always uses the library's `std` I/O
path.

## Configuration caveats

Configuration fields fail closed when their requested behavior is unavailable:

- `GGUFReaderConfig::buffer_size` bounds each temporary chunk while tensor
  bytes are accumulated; `use_mmap = true` returns `FeatureUnavailable` rather
  than silently using ordinary I/O;
- `StreamReaderConfig::buffer_size` serves the same bounded-read role;
  `validate_checksums = true` returns `FeatureUnavailable` because GGUF does
  not carry a checksum this reader can verify;
- `GGUFWriterConfig::buffer_size` remains a reserved hint because buffering is
  delegated to the supplied `Write`; `compress_metadata = true` returns
  `FeatureUnavailable` because GGUF v3 defines no compressed metadata form;
- file and stream writer `validate_data` fields do not weaken mandatory
  descriptor, offset, alignment, and exact payload-length checks. No additional
  optional semantic validation is currently attached to the flag.

`GGUFWriterConfig::compute_checksums` returns a non-cryptographic checksum in
each tensor's `WriteResult`; it does not embed that checksum in the GGUF file.
`StreamWriterConfig::buffer_size` is used by chunked writes. Its
`validate_data` field likewise does not disable mandatory format checks.

Do not treat a configuration field as a security control unless the selected
API path documents and tests that behavior.
