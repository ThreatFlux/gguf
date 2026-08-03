# Pull request

## Summary

Describe the problem and the approach taken.

## Validation

List the exact commands and platforms used to validate the change.

## Compatibility and safety

- [ ] Public API and serialized-format compatibility were considered.
- [ ] `no_std` and optional-feature behavior were considered where relevant.
- [ ] New or changed unsafe code includes documented invariants and tests.
- [ ] Untrusted GGUF input paths remain bounded and return structured errors.

## Checklist

- [ ] Tests cover the changed behavior.
- [ ] User-facing documentation and examples are updated.
- [ ] `cargo fmt --all -- --check` passes.
- [ ] Workspace Clippy passes with warnings denied.
- [ ] `cargo test --locked --workspace --all-features` passes.
