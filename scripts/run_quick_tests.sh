#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

echo "Checking formatting"
cargo fmt --all -- --check

echo "Checking the default library"
cargo check --locked -p gguf-rs-lib

echo "Running library and standalone integration tests"
cargo test --locked -p gguf-rs-lib --lib
cargo test --locked -p gguf-rs-lib --test integration_tests

echo "Checking documentation and package contents"
python3 scripts/check_docs.py
scripts/check_package.sh

echo "Quick checks passed"
