#!/bin/bash
set -e

echo "Running basic GGUF tests..."

# Just check compilation first
echo "Checking compilation..."
cargo check

# Run a single simple test
echo "Running format constants test..."
cargo test lib::tests::test_constants --lib

echo "Basic test completed!"