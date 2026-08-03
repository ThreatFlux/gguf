#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

if ! cargo llvm-cov --version >/dev/null 2>&1; then
    echo "cargo-llvm-cov is required; install it with:" >&2
    echo "  cargo install cargo-llvm-cov --locked" >&2
    exit 1
fi

mkdir -p target/coverage/html

echo "Running workspace tests with coverage instrumentation"
cargo llvm-cov \
    --locked \
    --workspace \
    --all-features \
    --no-report

echo "Writing LCOV and HTML reports"
cargo llvm-cov report \
    --locked \
    --lcov \
    --output-path target/coverage/lcov.info
cargo llvm-cov report \
    --locked \
    --html \
    --output-dir target/coverage/html

echo "Coverage reports:"
echo "  target/coverage/lcov.info"
echo "  target/coverage/html/index.html"
