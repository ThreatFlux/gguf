#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

if ! command -v rustup >/dev/null 2>&1; then
    echo "rustup is required: https://rustup.rs/" >&2
    exit 1
fi

echo "Installing formatter and linter components for the active toolchain"
rustup component add rustfmt clippy

echo "Fetching locked workspace dependencies"
cargo fetch --locked

echo "Checking the default library and CLI"
cargo check --locked -p gguf-rs-lib
cargo check --locked -p gguf-cli

echo "Development setup complete"
echo "Run scripts/run_quick_tests.sh for the short validation loop."
echo "Optional tools:"
echo "  cargo install cargo-audit --locked"
echo "  cargo install cargo-llvm-cov --locked"
