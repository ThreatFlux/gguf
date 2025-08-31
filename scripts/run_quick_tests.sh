#!/bin/bash
set -e

echo "Running quick GGUF tests..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Run unit tests
print_status "Running unit tests..."
cargo test unit -- --nocapture

# Run a subset of integration tests  
print_status "Running integration tests..."
cargo test integration::end_to_end_tests::test_complete_workflow -- --nocapture
cargo test integration::format_conversion_tests::test_round_trip_data_integrity -- --nocapture

# Run basic property tests with fewer iterations
print_status "Running property-based tests (reduced iterations)..."
PROPTEST_CASES=10 cargo test property_based -- --nocapture

# Run library tests
print_status "Running library tests..."
cargo test --lib -- --nocapture

print_status "Quick tests completed successfully!"