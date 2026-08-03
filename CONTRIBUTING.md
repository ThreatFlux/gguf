# Contributing to gguf-rs

Thank you for improving `gguf-rs-lib` or the workspace `gguf-cli`. Focused
changes with reproducible tests are the easiest to review.

## Before starting

- Search [existing issues](https://github.com/ThreatFlux/gguf/issues) and pull
  requests.
- Open an issue before a large API change, new format version, or new
  quantization implementation.
- Do not publish model files, credentials, private datasets, or generated
  coverage output in a pull request.
- Report security problems privately through
  [GitHub security advisories](https://github.com/ThreatFlux/gguf/security/advisories/new).

## Development setup

Install Rust with [rustup](https://rustup.rs/) and add the formatter and linter:

```bash
rustup component add rustfmt clippy

git clone https://github.com/ThreatFlux/gguf.git
cd gguf
cargo fetch --locked
```

The workspace tracks `Cargo.lock` because it contains an application. Do not
delete or ignore it. If an intentional dependency change updates the lockfile,
include that update in the same pull request.

The package's declared `rust-version` and the CI matrix define supported
compiler versions. New code must compile on the declared minimum version, not
only on the latest stable toolchain.

## Make a change

1. Create a focused branch from the current default branch.
2. Add or update tests that demonstrate the behavior.
3. Update public API docs, examples, and guides in the same change.
4. Run the checks below.
5. Describe compatibility, safety, and performance implications in the pull
   request.

Keep unrelated formatting or refactors out of a behavioral fix. Preserve
backward compatibility unless a breaking change has been discussed and is
intentional.

## Required local checks

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
cargo test --locked -p gguf-rs-lib --doc --all-features
cargo check --locked -p gguf-rs-lib --no-default-features --features alloc
cargo build --locked -p gguf-rs-lib --examples --all-features
python3 scripts/check_docs.py
./scripts/check_package.sh
```

`./scripts/run_quick_tests.sh` runs the shorter contributor loop.
`./scripts/test-all.sh` runs the full repository check set. See the
[testing guide](TESTING_GUIDE.md) for targeted commands and coverage.

## Code and API expectations

- Format with the repository's `rustfmt` configuration.
- Treat Clippy warnings as errors.
- Document every public API and its error conditions.
- Prefer checked arithmetic and bounded allocation when parsing file-controlled
  lengths, counts, dimensions, or offsets.
- Avoid panics on malformed input. Convenience APIs that can panic must say so
  in their documentation.
- Keep unsafe code narrowly scoped, justify its invariants with a `// SAFETY:`
  comment, and add boundary tests.
- Do not claim support for a GGUF type until descriptor parsing, exact payload
  sizing, and round-trip compatibility have been distinguished and tested.

## Format changes

A change that adds a GGUF version or tensor type should include:

- the authoritative upstream specification or implementation reference;
- parser tests for valid, truncated, malformed, and unsupported input;
- writer and independent-reader round-trip tests when writing is supported;
- byte-order and alignment behavior;
- exact block layout and size calculations for payload support;
- updates to [format support](docs/format-support.md).

Recognizing a tensor type ID is not the same as implementing its quantization
codec. Keep those claims separate in code, tests, and documentation.

## Testing expectations

Place focused unit tests with the relevant module or under `tests/unit/`.
Cross-module workflows belong under `tests/integration/`. Use property tests
for invariants over broad input spaces and keep regression seeds under
`proptest-regressions/`.

Tests must be:

- deterministic and independent;
- bounded in memory and runtime;
- explicit about the feature set they require;
- self-contained, using generated data or small reviewed fixtures;
- free of dependencies on private or machine-local model paths.

Do not check in generated GGUF files from examples. If a binary fixture is
necessary, place the smallest possible reviewed file under `tests/fixtures/`,
document how it was produced, and ensure a test actually consumes it.

## Documentation and examples

Examples must build in the feature combinations they advertise and return a
failure status when their operation fails. Avoid hardcoded model names or local
`data/` paths.

Run `python3 scripts/check_docs.py` after editing Markdown. The checker verifies
relative links and rejects retired repository identifiers and stale example
dependency versions.

Performance claims require a reproducible benchmark, environment details, and
a baseline that measures the same operation. Prefer a measured number with
scope over words such as “fast” or “zero-copy.”

## Package contents

`gguf-rs-lib` uses an explicit package allowlist. Before changing it or adding
new top-level content, run:

```bash
./scripts/check_package.sh
```

The package should contain the manifest, library source, license, README,
changelog, public guides, and useful examples. It must not contain workflows,
coverage reports, repository-only test output, local model files, or
development scratchpads.

`gguf-cli` is workspace-only and is not published separately.

## Pull request checklist

- [ ] The change is focused and its motivation is clear.
- [ ] User-visible behavior has tests.
- [ ] Public docs and examples match the implementation.
- [ ] Supported feature combinations compile.
- [ ] Format, Clippy, tests, docs, examples, and package checks pass.
- [ ] `Cargo.lock` is unchanged unless dependency resolution intentionally
      changed.
- [ ] No generated binaries, coverage files, secrets, or large model data are
      included.
- [ ] Compatibility, unsafe-code, allocation, and performance effects are
      described.

Maintainers may ask for commits to be reorganized, but readable history matters
more than a mechanically enforced commit-message convention.

## License

By contributing, you agree that your contribution is licensed under the
repository's [MIT License](LICENSE).
