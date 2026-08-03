# Safety and validation

GGUF files are structured binary input. Treat files from outside your trust
boundary as untrusted even when the crate or CLI reports that parsing
succeeded.

## What the synchronous reader checks

`GGUFFileReader::new` uses the default configuration and checks:

- the exact `GGUF` magic value;
- version 3;
- no more than 1,000,000 tensor descriptors;
- no more than 100,000 metadata entries;
- metadata keys up to 65,535 bytes with exact dot-separated
  `lower_snake_case` ASCII hierarchy validation, incremental string/array
  allocation, and array nesting up to 64 levels;
- independent 256 MiB budgets for serialized metadata bytes and estimated
  decoded metadata allocations; the second limit prevents compact arrays from
  expanding into multi-gigabyte `MetadataValue` vectors;
- recognized metadata and tensor type IDs;
- boolean encodings restricted to 0 or 1;
- UTF-8 for names, keys, and string values;
- tensor names up to 64 bytes and exactly 1–4 dimensions per descriptor;
- tensor name, shape, and descriptor consistency;
- exact block-width and byte-size validation for every recognized quantized
  tensor type;
- alignment and overlap between calculated tensor payload ranges;
- arithmetic overflow and whether every declared payload range fits the input.

Both reader configurations expose separate serialized metadata
(`max_metadata_size`) and decoded allocation (`max_decoded_metadata_size`)
limits. The stream reader also has a configurable tensor-count policy. The
seekable file reader can enforce a nonzero `GGUFReaderConfig::max_file_size`;
its default file-size policy is unlimited.

## What it does not establish

A successful parse is not proof that a file is safe or meaningful. In
particular, default parsing does not establish:

- that a model architecture recognizes otherwise syntactically valid metadata
  keys and tensor names;
- that payload bytes encode valid numeric values;
- cryptographic integrity, authenticity, provenance, or freedom from malicious
  model behavior;
- compatibility with another GGUF implementation.

The parser validates raw byte ranges for supported tensor layouts without
loading them. Use eager loading only when its memory cost is acceptable. For
high-assurance ingestion, apply an application-level file-size limit,
cryptographic digest or signature policy, and independent semantic checks.

The CLI's `validate` command applies `GGUFFileReader::new` to one file, direct
`.gguf` children of a directory, or all `.gguf` descendants when
`--recursive` is set. `--integrity` additionally reads every declared tensor
payload. Any invalid candidate, traversal error, or directory with no matching
files produces a nonzero exit status. Payload reads reuse a bounded buffer and
are not retained, but they are not hashes: validation still does not
authenticate publishers or detect content changes that remain structurally
valid.

The CLI's `compare` command checks GGUF version, metadata values, and tensor
type, shape, relative offset, and expected payload size. `--data` also compares
the exact payload bytes of tensors with matching descriptors through bounded
buffers without retaining the payloads. It does not establish numeric or
model-level correctness.

Complete file and stream writes validate descriptors, unique names, and
payload lengths before emitting the header, and flush before returning
success. Path-based helpers perform the same preflight before they open and
truncate the destination. Those checks prevent invalid caller input from
producing a partial file, but an underlying I/O failure can still interrupt
output. After any write or flush error, discard the writer and destination and
retry against a fresh target; do not resume a failed GGUF write in place.

## Checksums

`TensorData::checksum` is a small non-cryptographic checksum intended for
accidental corruption checks. It is not collision resistant and must not be
used as a security control.

`TensorReadOptions::compute_checksum = true` computes that value for the bytes
just read. GGUF supplies no reference checksum, so the result is for comparison
against a separately trusted value; the option does not itself validate
integrity or authenticity.

`GGUFWriterConfig::compute_checksums = true` returns a non-cryptographic
checksum in each tensor's `WriteResult`; it does not persist that checksum in
the GGUF file. GGUF has no standard checksum field, so
`StreamReaderConfig::validate_checksums = true` returns
`GGUFError::FeatureUnavailable` instead of implying that verification
occurred. Use a standard digest such as SHA-256 outside this crate when
integrity is security relevant.

## Memory mapping

The default feature set contains no executed `unsafe` block. Enabling `mmap`
uses the `memmap2` mapping boundary. `MmapGGUFFile::mmap` is an unsafe
constructor because memory-map soundness depends on external file behavior
that a safe wrapper cannot enforce. `MmapGGUFFile::from_mmap` accepts an
already-established mapping.

While a map or mapped tensor view is alive:

- do not truncate or otherwise mutate the underlying file;
- avoid mapping files controlled by a concurrently running untrusted process;
- validate offsets and lengths before creating mapped tensor data;
- remember that a mapped slice can fault later when pages are accessed.

`MmapGGUFFile` currently validates only magic and version. It is not a
zero-copy replacement for the complete synchronous parser.

## Allocation and denial of service

Count, element, nesting, name, and aggregate metadata limits reduce oversized
allocations but do not make resource use constant. A structurally valid file
can still be large, and loading payloads can allocate up to the declared,
validated tensor size.

For untrusted input:

1. reject files above an application-specific byte limit before parsing;
2. parse in a process with memory and time limits when practical;
3. avoid eager payload loading unless required;
4. cap downstream decompression, conversion, and tensor processing separately;
5. fail closed on unsupported tensor IDs, byte order, or alignment.

## Validation-bypassing helpers

Some constructors, such as `TensorShape::new_unchecked`, bypass normal policy
checks even though they are safe Rust functions. Reserve them for values whose
invariants have already been established.

`TensorData::new_borrowed` accepts only a `'static` slice. It does not create a
borrow tied to an arbitrary input buffer.

## Reporting a vulnerability

Use
[GitHub private vulnerability reporting](https://github.com/ThreatFlux/gguf/security/advisories/new)
for suspected vulnerabilities. Do not publish exploit details in a public
issue before maintainers have had an opportunity to investigate.
