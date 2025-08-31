#!/bin/bash
set -euo pipefail

# Comprehensive test script for gguf_rs

echo "🧪 Running comprehensive test suite for gguf_rs..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        return 1
    fi
}

# Track overall status
OVERALL_STATUS=0

echo "🔍 Running basic checks..."

# Check formatting
echo -n "Checking code formatting... "
if cargo fmt --check &> /dev/null; then
    print_status 0 "Code formatting"
else
    print_status 1 "Code formatting (run 'cargo fmt' to fix)"
    OVERALL_STATUS=1
fi

# Run Clippy
echo -n "Running Clippy linter... "
if cargo clippy --all-targets --all-features -- -D warnings &> /dev/null; then
    print_status 0 "Clippy linter"
else
    print_status 1 "Clippy linter"
    OVERALL_STATUS=1
fi

echo ""
echo "🏗️ Building and testing..."

# Test different feature combinations
FEATURE_COMBINATIONS=(
    ""
    "--features async"
    "--features mmap"
    "--features async,mmap"
    "--no-default-features"
)

for features in "${FEATURE_COMBINATIONS[@]}"; do
    echo -n "Testing with features: '${features:-default}' ... "
    if cargo test $features --quiet &> /dev/null; then
        print_status 0 "Tests (${features:-default})"
    else
        print_status 1 "Tests (${features:-default})"
        OVERALL_STATUS=1
    fi
done

# Test CLI
echo -n "Building CLI... "
if cargo build -p gguf-cli --quiet &> /dev/null; then
    print_status 0 "CLI build"
    
    echo -n "Testing CLI functionality... "
    if ./target/debug/gguf-cli --help &> /dev/null; then
        print_status 0 "CLI help"
    else
        print_status 1 "CLI help"
        OVERALL_STATUS=1
    fi
else
    print_status 1 "CLI build"
    OVERALL_STATUS=1
fi

echo ""
echo "📚 Testing documentation..."

# Build documentation
echo -n "Building documentation... "
if cargo doc --all-features --no-deps --quiet &> /dev/null; then
    print_status 0 "Documentation build"
else
    print_status 1 "Documentation build"
    OVERALL_STATUS=1
fi

# Test examples
echo -n "Testing examples... "
if cargo build --examples --all-features --quiet &> /dev/null; then
    print_status 0 "Examples build"
else
    print_status 1 "Examples build"
    OVERALL_STATUS=1
fi

echo ""
echo "⚡ Performance tests..."

# Run benchmarks (don't fail on benchmark issues)
echo -n "Running benchmarks... "
if cargo bench --all-features --quiet &> /dev/null; then
    print_status 0 "Benchmarks"
else
    echo -e "${YELLOW}⚠️ Benchmarks (non-critical)${NC}"
fi

echo ""
echo "🔒 Security checks..."

# Security audit
if command -v cargo-audit &> /dev/null; then
    echo -n "Running security audit... "
    if cargo audit --quiet &> /dev/null; then
        print_status 0 "Security audit"
    else
        print_status 1 "Security audit"
        OVERALL_STATUS=1
    fi
else
    echo -e "${YELLOW}⚠️ cargo-audit not installed, skipping security check${NC}"
fi

echo ""
echo "📊 Test coverage..."

# Code coverage (Linux only, optional)
if command -v cargo-tarpaulin &> /dev/null && [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -n "Running code coverage analysis... "
    if cargo tarpaulin --all-features --ignore-tests --quiet &> /dev/null; then
        print_status 0 "Code coverage"
    else
        echo -e "${YELLOW}⚠️ Code coverage (non-critical)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ cargo-tarpaulin not available, skipping coverage${NC}"
fi

echo ""
echo "🎯 Final status..."

if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed! Ready for deployment.${NC}"
else
    echo -e "${RED}💥 Some tests failed. Please fix the issues above.${NC}"
fi

exit $OVERALL_STATUS