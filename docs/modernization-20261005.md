# Rust and dependency modernization — 2026-10-05

Development, release verification and stable CI now use Rust 1.99.0, released
on [2026-10-01](https://blog.rust-lang.org/2026/10/01/Rust-1.99.0/).
The library and workspace CLI retain their declared Rust 1.87 minimum;
the dedicated CI lane selects that compiler explicitly despite the development
toolchain file. Windows, macOS and the existing informational beta lane retain
their previous test scope.

All 18 unique direct registry dependencies use the newest stable, non-yanked
release checked through the [crates.io API](https://crates.io/data-access).
The lockfile refresh updates 35 packages. The CLI's `serde_yaml` remains
0.9.34+deprecated, which is its newest stable release; replacing that backend
is separate from this update. Strict RustSec and dependency policy checks
continue to reject warnings and have no advisory exceptions.

The six workflows retain full commit SHA pins. Updated action releases include
Rust toolchain setup, CodeQL 4.38.2, installer 2.87.25, Codecov 7.1.1,
GitHub Pages deployment 5.0.1 and GitHub release creation 3.0.3.
Coverage now runs on pull requests and retains a required, nonempty LCOV
artifact. Codecov delivery and Pages deployment keep their existing main-only
conditions. Auto Release remains owned by the existing ThreatFlux reusable
workflow at version 0.7.5. This update does not invoke release or deployment.

`make hooks-install` installs a worktree-specific pre-push hook. `make ci-local`
runs formatting, strict Clippy, workspace and documentation tests, feature
checks including the alloc-only API test, documentation contracts, package
contents, workflow validation, example builds and archive verification.

Package versions, editions, library exports, CLI commands and feature contracts
remain unchanged. No container definitions exist in this repository.
