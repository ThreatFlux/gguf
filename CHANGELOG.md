# Changelog

All notable user-facing changes to this project are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-08-03

Version `0.2.6` was not published, but its public tag is immutable and will not
be moved or reused.

### Added

- Exact checked size calculations and block-divisibility validation for the
  supported GGML quantized tensor layouts.
- Current GGML storage geometry for `TQ1_0`, `TQ2_0`, `MXFP4`, `NVFP4`,
  `Q1_0`, and `Q2_0`.
- Bounded metadata parsing, exact hierarchical key validation, nesting and
  length policies, checked arithmetic, canonical 1–4-dimensional tensor
  descriptors, tensor range checks, duplicate-name rejection, and alignment
  validation.
- Format, feature, CLI, example, safety, testing, contribution, and package
  documentation grounded in the implemented API.
- Documentation-link and crate-package-content checks.
- A tracked workspace `Cargo.lock` and an explicit library package allowlist.
- Process-level CLI coverage for inspection, all metadata output formats,
  recursive validation, integrity reads, and structural or payload comparison.

### Changed

- Corrected canonical GGUF tensor identifiers while retaining removed
  SDK-only enum variants solely for source compatibility.
- Replaced estimated quantized payload sizing with exact block widths and byte
  sizes for every recognized GGML and IQ layout.
- Made `GGUFTensorType::calculate_size` and
  `QuantizationParams::calculate_storage_size` return `Option<u64>` instead of
  using `u64::MAX` as an ambiguous unsupported/overflow sentinel.
- Honored `general.alignment` when reading seekable and streaming GGUF files.
- Updated dependencies, declared Rust 1.87 as the minimum supported version,
  and marked `gguf-cli` as a workspace-only package.
- Hardened CI, security, dependency-update, and release workflows with locked
  dependencies, least-privilege permissions, pinned actions, and package
  verification.
- Replaced stale API snippets, repository names, version pins, performance
  claims, and hardcoded local-model examples.
- Made generated examples write below `target/` and made coverage reports write
  below `target/coverage/`.
- Implemented detailed tensor inspection, JSON/YAML/TOML/table metadata output,
  recursive directory validation, payload reads, and file comparison in
  `gguf-cli`; errors and differences now return nonzero status.
- Made CLI metadata comparison exact for nested non-finite floating-point
  values, avoiding false differences when identical NaN bit patterns recur.
- Changed unsupported memory-mapping, stream-checksum, and metadata-compression
  configuration requests to return `FeatureUnavailable` instead of being
  silently ignored; documented file-writer buffering as a reserved hint.
- Enforced header-declared counts and section order in low-level file and
  stream writers, including alignment padding for every tensor.
- Made complete file and stream writes flush their underlying writer before
  reporting success, so buffered I/O errors are not deferred to drop.
- Made complete writers validate every descriptor, unique tensor name, and
  payload before emitting the header, avoiding partial output for invalid input.
- Made path-based writer and builder helpers finish that preflight before
  opening and truncating their destination.
- Made tensor reads reject invalid alignment and unavailable decompression
  before consuming bytes or allocating the declared payload.
- Replaced the misleading `TensorReadOptions::validate_integrity` and
  `TensorReadResult::was_validated` fields with `compute_checksum` and the
  existing optional checksum result; GGUF provides no reference checksum for
  this helper to validate.
- Renamed `TensorMemoryInfo::compression_ratio` and
  `GGUFMemoryUsage::compression_ratio()` to `loaded_fraction` to match their
  loaded-bytes / expected-bytes meaning.
- Made `add_f32_tensor`, `add_i32_tensor`, and `add_quantized_tensor` return
  `Result<GGUFBuilder>` so invalid shapes, block geometry, and payload lengths
  fail immediately; quantized builders now declare and validate
  `general.quantization_version = 2`.
- Standardized `TensorShape` on GGML descriptor order (dimension 0 contiguous),
  including matrix, broadcasting, and matrix-multiplication helpers, and
  replaced panic-prone conversions and unchecked unsqueeze operations with
  fallible APIs.
- Removed the nonstandard 65,536-element/string metadata limits. Metadata
  decoding now grows fallibly in fixed increments and applies independent
  serialized-byte and decoded-allocation budgets to prevent compact-array
  memory amplification.
- Unified `gguf_rs_lib::metadata` with the canonical format metadata types;
  the crate no longer exposes a second, incompatible metadata collection.

### Migration from 0.2.x

`GGUFTensorType` is `#[repr(u32)]`; correcting its public discriminants to the
canonical GGUF IDs is therefore a breaking API change and requires 0.3.0.
Code that casts enum variants to integers, matches raw IDs, or persists those
integers must use this mapping:

| Variant | 0.2.x discriminant | 0.3.0 discriminant |
| --- | ---: | ---: |
| `I8` | 30 | 24 |
| `I16` | 31 | 25 |
| `I32` | 24 | 26 |
| `I64` | 25 | 27 |
| `F64` | 26 | 28 |
| `IQ1_M` | 27 | 29 |
| `BF16` | 28 | 30 |
| `IQ4_UNI` | 29 | `u32::MAX` |

`IQ4_UNI` is retained only as an SDK source-compatibility variant and is
rejected for GGUF file I/O. Removed IDs 4 (`Q4_2`) and 5 (`Q4_3`) are likewise
rejected rather than decoded. Regenerate any application-owned numeric lookup
tables and do not reinterpret old SDK discriminants as canonical file IDs.

The typed tensor convenience builders now return `Result<GGUFBuilder>`; add
`?`, `unwrap`, or explicit error handling after each call in a chain.
`GGUFBuilder::language_model` now accepts `u32` context and embedding lengths.
`MetadataBuilder::with_llama_params` now accepts only `u32` context and
embedding lengths; remove the former vocabulary-size argument and add any
architecture-specific vocabulary metadata explicitly.

`GGUFTensorType::element_size` now returns `Option<usize>` because quantized
formats are block-addressed; use `block_size` and `block_size_bytes` for their
physical geometry. `TensorInfo::calculate_layout` now returns `Result`, reports
GGML byte strides with dimension 0 contiguous, and identifies the result as
`MemoryLayout::Ggml`.

`TensorShape` conversion from vectors and slices now uses `TryFrom`, and
`unsqueeze_front` / `unsqueeze_back` return `Result<TensorShape>`. Matrix
helpers expose GGML order: a logical `rows × cols` matrix is stored as
`[cols, rows]`.

`gguf_rs_lib::metadata::Metadata` now refers to the same validated type as
`gguf_rs_lib::format::Metadata`. Code that accessed the old compatibility
type's public `entries` field should use `insert`, `get`, and the iterator APIs.

Reader configuration struct literals must add
`max_decoded_metadata_size` (and `max_metadata_size` for
`GGUFReaderConfig`) or use `..Default::default()`. The two budgets separately
limit serialized metadata bytes and estimated decoded allocations.

`TensorMemoryInfo::compression_ratio` and
`GGUFMemoryUsage::compression_ratio()` are now `loaded_fraction`; neither
metric described encoded tensor compression. Enabling `mmap` also now requires
an explicit `unsafe` call to `MmapGGUFFile::mmap` and upholding its documented
file-stability contract.

`TensorReadOptions::validate_integrity` is now `compute_checksum`, and
`TensorReadResult::was_validated` was removed. A returned checksum is a
non-cryptographic value for comparison with a separately trusted reference;
it is not evidence that the payload was authenticated or validated.

`GGUFTensorType::calculate_size` and
`QuantizationParams::calculate_storage_size` now return `Option<u64>`; handle
`None` for unsupported types or arithmetic overflow. The existing
`checked_calculate_size` helper remains available.

### Removed

- The tracked generated `lcov.info` report and invalid root `test_model.gguf`
  artifact.
- Vendor-specific real-model examples that depended on untracked local files.
- Unused CLI dependencies and unsupported manifest feature declarations.
- Speculative quantization-quality, dynamic-range, losslessness,
  inference-speed, precision, model-selection, and “modern” classification
  helpers, an ungrounded removed-type replacement recommendation, and
  ambiguous nominal-bit-width fields; the crate reports exact physical storage
  geometry but does not rank numeric quality or runtime behavior without codecs
  and empirical model data.

## [0.2.5] - 2025-09-02

Version 0.2.5 is the latest release of `gguf-rs-lib` published to crates.io
before the changes collected in this changelog. The repository did not contain
curated historical release notes, so no earlier change details are inferred
here.

[Unreleased]: https://github.com/ThreatFlux/gguf/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ThreatFlux/gguf/compare/v0.2.5...v0.3.0
[0.2.5]: https://github.com/ThreatFlux/gguf/releases/tag/v0.2.5
