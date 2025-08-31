#!/bin/bash
set -e

echo "Running GGUF tests with coverage measurement..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if cargo-tarpaulin is installed
if ! command -v cargo-tarpaulin &> /dev/null; then
    print_warning "cargo-tarpaulin not found, installing..."
    cargo install cargo-tarpaulin
fi

# Clean previous coverage data
print_status "Cleaning previous coverage data..."
cargo clean

# Create coverage output directory
mkdir -p target/coverage

print_status "Running tests with coverage measurement..."

# Run tests with tarpaulin
cargo tarpaulin \
    --verbose \
    --all-features \
    --workspace \
    --timeout 120 \
    --out Html \
    --out Xml \
    --output-dir target/coverage \
    --exclude-files "benches/*" \
    --exclude-files "examples/*" \
    --exclude-files "target/*" \
    --exclude-files "tests/*" \
    --line \
    --branch \
    --count

COVERAGE_EXIT_CODE=$?

if [ $COVERAGE_EXIT_CODE -eq 0 ]; then
    print_status "Coverage analysis completed successfully!"
    
    # Extract coverage percentage from the output
    if [ -f "target/coverage/cobertura.xml" ]; then
        COVERAGE=$(grep -o 'line-rate="[^"]*"' target/coverage/cobertura.xml | head -1 | cut -d'"' -f2)
        COVERAGE_PERCENT=$(echo "$COVERAGE * 100" | bc -l | cut -d'.' -f1)
        
        echo ""
        echo "======================================"
        echo "  COVERAGE RESULTS"
        echo "======================================"
        echo "Total Line Coverage: ${COVERAGE_PERCENT}%"
        
        if [ "$COVERAGE_PERCENT" -ge 90 ]; then
            print_status "✅ Coverage goal achieved (≥90%)!"
        else
            print_warning "⚠️  Coverage below target (${COVERAGE_PERCENT}% < 90%)"
        fi
        
        echo ""
        echo "Coverage reports generated:"
        echo "  HTML: target/coverage/tarpaulin-report.html"
        echo "  XML:  target/coverage/cobertura.xml"
        echo ""
    fi
else
    print_error "Coverage analysis failed with exit code $COVERAGE_EXIT_CODE"
    exit $COVERAGE_EXIT_CODE
fi

# Also run a separate test for ignored tests or specific features
print_status "Running additional tests..."

# Run property-based tests with more iterations
cargo test --test property_based -- --ignored --nocapture

# Run integration tests separately
cargo test --test integration -- --nocapture

print_status "All tests completed!"

# Open coverage report if in interactive mode
if [ -t 1 ] && command -v xdg-open &> /dev/null; then
    echo "Would you like to open the coverage report? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        xdg-open target/coverage/tarpaulin-report.html
    fi
elif [ -t 1 ] && command -v open &> /dev/null; then
    echo "Would you like to open the coverage report? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        open target/coverage/tarpaulin-report.html
    fi
fi