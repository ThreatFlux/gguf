# CLI guide

`gguf-cli` is an unpublished workspace application for inspecting, validating,
and comparing GGUF files. The crates.io package named `gguf` is unrelated, so
do not install it expecting this CLI.

## Install from source

```bash
git clone https://github.com/ThreatFlux/gguf.git
cd gguf
cargo install --locked --path gguf-cli
```

For development, replace `gguf-cli` with
`cargo run --locked -p gguf-cli --` in the commands below. `--verbose` prints
progress details, and `--no-color` disables colored validation labels; both are
global options. Progress is written to standard output, so omit `--verbose`
when another program consumes JSON, YAML, or TOML output.

## File information

```bash
gguf-cli info model.gguf
gguf-cli info model.gguf --detailed
```

The command uses the complete synchronous parser and prints the GGUF version,
tensor and metadata counts, and tensor alignment. `--detailed` adds the file
size, checked total tensor payload size, and `general.name` or
`general.architecture` values when present.

## Tensor descriptors

```bash
gguf-cli tensors model.gguf
gguf-cli tensors model.gguf --summary
gguf-cli tensors model.gguf --filter attention
```

The default view prints each matching tensor's name, recognized type, shape,
element count, checked payload size, and relative data offset. `--summary`
prints one compact line per tensor. `--filter` performs a case-sensitive
substring match on tensor names and works with either view.

## Metadata

```bash
gguf-cli metadata model.gguf
gguf-cli metadata model.gguf --format json
gguf-cli metadata model.gguf --format yaml --key general.
gguf-cli metadata model.gguf --format toml
```

Supported formats are `table`, `json`, `yaml`, and `toml`; any other value is
rejected during argument parsing. Output keys are sorted deterministically.
JSON, YAML, and TOML emit native scalars and arrays rather than Rust enum tags.
The default `table` format is line-oriented `key: value` text. `--key` performs
a case-sensitive substring match. If a metadata value cannot be represented by
the selected serializer, conversion fails with a nonzero status instead of
falling back to another format. In particular, JSON and TOML reject non-finite
floating-point values rather than silently converting them to `null`.

## Validation

```bash
gguf-cli validate model.gguf
gguf-cli validate model.gguf --integrity
gguf-cli validate models/
gguf-cli validate models/ --recursive --integrity
```

For one file, validation succeeds only when the complete synchronous parser
accepts its header, metadata, tensor descriptors, and declared payload ranges.
`--integrity` additionally reads every declared tensor payload through one
reused bounded buffer without retaining model data.

For a directory, the command validates direct `.gguf` children, or all `.gguf`
descendants with `--recursive`. Extension matching is case-insensitive. Files
are processed in sorted path order and receive individual `VALID` or `INVALID`
lines. The command returns nonzero if any candidate is invalid, directory
traversal fails, or no matching files are found. Supplying `--recursive` with a
file is also an error.

Validation does not compute a cryptographic digest, authenticate the file's
publisher, or establish that tensor bytes are numerically meaningful for a
model architecture.

## Comparison

```bash
gguf-cli compare baseline.gguf candidate.gguf
gguf-cli compare baseline.gguf candidate.gguf --data
```

The default comparison checks the GGUF version, every metadata key and value,
and every tensor descriptor's type, dimensions, relative offset, and checked
payload size. `--data` also reads and compares exact payload bytes for tensors
whose descriptors match, reusing bounded buffers instead of retaining whole
tensor payloads. Differences are printed as `DIFF:` lines and produce a nonzero
exit status. String and array differences are summarized instead of printing
potentially enormous metadata values. Without `--data`, files with the same
structure but different payload bytes compare successfully by design.

## Exit behavior

Successful commands return zero. Invalid command-line values, file and
directory errors, parse or payload-read failures, validation failures, and
comparison differences return nonzero. Diagnostics are written to standard
error; command results and per-file validation lines are written to standard
output.
