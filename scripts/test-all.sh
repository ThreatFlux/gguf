#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

echo "Checking formatting"
cargo fmt --all -- --check

echo "Running Clippy"
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings

echo "Running workspace tests"
cargo test --locked --workspace --all-features

echo "Checking no_std + alloc"
cargo check --locked -p gguf-rs-lib --no-default-features --features alloc
cargo test --locked -p gguf-rs-lib --no-default-features --features alloc --test alloc_only

echo "Testing and building documentation"
cargo test --locked -p gguf-rs-lib --doc --all-features
cargo doc --locked --workspace --all-features --no-deps

echo "Building examples"
cargo build --locked -p gguf-rs-lib --examples --all-features

echo "Checking repository documentation and package contents"
python3 scripts/check_docs.py
scripts/check_package.sh

echo "All checks passed"
