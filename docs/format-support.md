# Format and type support

This page distinguishes descriptor recognition from complete tensor-payload
support. That distinction matters for GGUF because a parser can understand a
type ID without implementing that type's block layout or codec.

The upstream reference is the
[GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
Upstream can add tensor types without changing the GGUF container version, so
container-version support alone does not imply compatibility with every model.

## Container version and byte order

The current reader and writer support only GGUF version 3. Versions 1, 2, and
values newer than 3 return `GGUFError::UnsupportedVersion`.

All numeric fields are currently read and written as little-endian. GGUF v3
allows big-endian files, but this crate does not detect or parse them.

The synchronous and streaming readers honor a valid `general.alignment`
metadata value and use 32 bytes when it is absent. Tensor offsets are checked
against the selected alignment.

## Metadata

The parser recognizes all metadata value identifiers defined by GGUF IDs 0–12:

- `U8`, `I8`, `U16`, `I16`, `U32`, and `I32`
- `F32` and `F64`
- `Bool` and UTF-8 `String`
- homogeneous `Array` values
- `U64` and `I64`

String and array lengths use the GGUF `u64` representation. The readers grow
their buffers incrementally instead of reserving a declared length up front;
the aggregate metadata budgets described below bound normal file and stream
parsing. Arrays may nest up to 64 levels as a library safety policy.

Metadata keys must be nonempty ASCII and no longer than 65,535 bytes. Readers,
writers, and builders enforce a dot-separated hierarchy whose components are
`lower_snake_case`: each dot component and underscore-separated word is
nonempty, and words contain only lowercase ASCII letters or digits. For
example, `general.name`, `tokenizer.ggml.model`, and `layer.0.name` are valid;
uppercase letters, hyphens, empty dot components, and leading, trailing, or
repeated underscores are rejected.

## Tensor descriptors

The parser accepts canonical tensor type IDs 0–3, 6–30, 34–35, and 39–42.
This table tracks the current upstream
[`ggml_type` registry](https://github.com/ggml-org/ggml/blob/master/include/ggml.h);
the enum embedded in `gguf.md` can lag the implementation headers:

| Group | Recognized names |
| --- | --- |
| Scalar/unquantized | `F32`, `F16`, `I32`, `I64`, `F64`, `BF16`, `I8`, `I16` |
| Block quantized | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q1_0`, `Q2_0` |
| K-quants | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` |
| IQ variants, IDs 16–20 | `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ1_S`, `IQ4_NL` |
| IQ variants, IDs 21–23 and 29 | `IQ3_S`, `IQ2_S`, `IQ4_XS`, `IQ1_M` |
| Ternary quantization | `TQ1_0`, `TQ2_0` |
| Microscaling quantization | `MXFP4`, `NVFP4` |

Removed IDs 4 (`Q4_2`) and 5 (`Q4_3`) are rejected. The public Rust enum keeps
those names, plus the historical SDK-only `IQ4_UNI` name, for source
compatibility; no GGUF integer maps to them. Unknown and newer IDs are also
rejected.

The parser accepts exactly 1–4 tensor dimensions. Each dimension is limited to
1,000,000,000 by the high-level tensor shape. A zero-sized dimension is
accepted by that shape validation, so callers must decide whether empty
tensors are valid for their application.

Descriptor dimensions use GGML order: dimension 0 is contiguous. Logical
element strides therefore grow from the front (`[2, 3, 4]` has strides
`[1, 2, 6]`). `TensorInfo::calculate_layout` reports GGML byte strides and uses
the encoded block size for quantized stride 0; it does not reinterpret shapes
as a host array library's row-major dimension order.

## Tensor payloads

For scalar types, payload byte counts are calculated directly from the element
count and these widths:

| IDs | Types | Bytes per element |
| --- | --- | ---: |
| 0, 26 | `F32`, `I32` | 4 |
| 1, 25, 30 | `F16`, `I16`, `BF16` | 2 |
| 24 | `I8` | 1 |
| 27, 28 | `I64`, `F64` | 8 |

For every accepted quantized type, payload sizing uses the exact GGML block
geometry below:

| ID | Type | Elements per block | Bytes per block |
| ---: | --- | ---: | ---: |
| 2 | `Q4_0` | 32 | 18 |
| 3 | `Q4_1` | 32 | 20 |
| 6 | `Q5_0` | 32 | 22 |
| 7 | `Q5_1` | 32 | 24 |
| 8 | `Q8_0` | 32 | 34 |
| 9 | `Q8_1` | 32 | 36 |
| 10 | `Q2_K` | 256 | 84 |
| 11 | `Q3_K` | 256 | 110 |
| 12 | `Q4_K` | 256 | 144 |
| 13 | `Q5_K` | 256 | 176 |
| 14 | `Q6_K` | 256 | 210 |
| 15 | `Q8_K` | 256 | 292 |
| 16 | `IQ2_XXS` | 256 | 66 |
| 17 | `IQ2_XS` | 256 | 74 |
| 18 | `IQ3_XXS` | 256 | 98 |
| 19 | `IQ1_S` | 256 | 50 |
| 20 | `IQ4_NL` | 32 | 18 |
| 21 | `IQ3_S` | 256 | 110 |
| 22 | `IQ2_S` | 256 | 82 |
| 23 | `IQ4_XS` | 256 | 136 |
| 29 | `IQ1_M` | 256 | 56 |
| 34 | `TQ1_0` | 256 | 54 |
| 35 | `TQ2_0` | 256 | 66 |
| 39 | `MXFP4` | 32 | 17 |
| 40 | `NVFP4` | 64 | 36 |
| 41 | `Q1_0` | 128 | 18 |
| 42 | `Q2_0` | 64 | 18 |

Tensor validation requires the first dimension to be divisible by the type's
block width, and checked APIs reject size overflow. Reader range checks and
writer payload-length checks use these exact raw sizes.

Recognition still is not a quantization codec: the crate does not quantize,
dequantize, or assess numeric quality. Callers that write a quantized tensor
must supply bytes already encoded in the named GGML block layout.

## Reading behavior

`GGUFFileReader::new`:

1. reads and validates the v3 header;
2. applies count limits;
3. parses metadata and tensor descriptors;
4. resolves `general.alignment`, defaulting to 32 bytes;
5. checks descriptor alignment, consistency, overlap, and file bounds using
   checked payload sizes.

It does not eagerly load payload bytes by default. `GGUFStreamReader` performs
the same general parse for a non-seekable source, with configurable metadata
size and tensor count limits. `GGUFReaderConfig::use_mmap = true` and
`StreamReaderConfig::validate_checksums = true` return
`GGUFError::FeatureUnavailable`; neither request is silently ignored.

The async and memory-map convenience file types are different: they currently
validate only the first eight bytes (magic and version). The async result has
empty metadata/tensor collections; the mapped view exposes the mapped bytes
and a bounded primitive reader but does not parse descriptors. See
[feature semantics](features.md).

## Writing behavior

The high-level `GGUFBuilder` and `GGUFFileWriter::write_complete_file` write
version 3 and compute tensor offsets relative to the start of the data section.
Low-level file and stream writers enforce header-declared metadata and tensor
counts, section order, and aligned tensor padding. Complete writes validate all
descriptors, tensor names, and payload lengths before emitting the header, then
flush the supplied writer before reporting success. Path-based helpers perform
that preflight before opening and truncating the destination. An underlying I/O
failure can still leave partial output; discard that writer and destination
and retry the complete write against a fresh target.

`GGUFWriterConfig::compress_metadata = true` returns
`GGUFError::FeatureUnavailable`, because GGUF v3 has no standard compressed
metadata representation. Its `buffer_size` is currently a reserved hint;
buffering remains the responsibility of the supplied writer.
`compute_checksums = true` returns each tensor's non-cryptographic checksum in
its `WriteResult` but does not store checksums in the GGUF file.

For interoperable output:

- use little-endian data;
- retain the 32-byte default alignment unless a custom value is required;
  complete-file and stream writers insert or verify matching `general.alignment`
  metadata, while manually sequenced low-level writes must supply it;
- supply already encoded blocks for quantized payload types;
- read the result back with an independent GGUF implementation when exchanging
  files across projects.

The crate stores raw bytes; it does not verify that bytes encode numerically
meaningful values for a model architecture.
