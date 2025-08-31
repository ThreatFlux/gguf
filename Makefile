.PHONY: all build test bench check fmt lint clean doc audit

# Default target
all: fmt check test

# Build the project
build:
	@echo "Building project..."
	@cargo build --all-features

# Build in release mode
release:
	@echo "Building release..."
	@cargo build --release --all-features

# Run tests
test:
	@echo "Running tests..."
	@cargo test --all-features

# Run benchmarks
bench:
	@echo "Running benchmarks..."
	@cargo bench --all-features

# Check code (compile without building)
check:
	@echo "Checking code..."
	@cargo check --all-features

# Format code
fmt:
	@echo "Formatting code..."
	@cargo fmt --all

# Lint with clippy
lint:
	@echo "Running clippy..."
	@cargo clippy --all-targets --all-features -- -D warnings

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@cargo clean

# Generate documentation
doc:
	@echo "Generating documentation..."
	@cargo doc --all-features --no-deps --open

# Security audit
audit:
	@echo "Running security audit..."
	@cargo audit

# Run all checks (for CI)
ci: fmt lint test doc

# Install development dependencies
setup:
	@echo "Installing development dependencies..."
	@rustup component add rustfmt clippy
	@cargo install cargo-audit

# Test with different feature combinations
test-features:
	@echo "Testing with no features..."
	@cargo test --no-default-features
	@echo "Testing with default features..."
	@cargo test
	@echo "Testing with all features..."
	@cargo test --all-features

# Run examples
examples:
	@echo "Running examples..."
	@cargo run --example create_test_gguf
	@cargo run --example inspect_gguf test_model.gguf || true

# Check MSRV (Minimum Supported Rust Version)
msrv:
	@echo "Checking MSRV (1.89.0)..."
	@cargo +1.89.0 check --all-features

help:
	@echo "Available targets:"
	@echo "  all         - Format, check, and test (default)"
	@echo "  build       - Build the project"
	@echo "  release     - Build in release mode"
	@echo "  test        - Run tests"
	@echo "  bench       - Run benchmarks"
	@echo "  check       - Check code compilation"
	@echo "  fmt         - Format code"
	@echo "  lint        - Run clippy linter"
	@echo "  clean       - Clean build artifacts"
	@echo "  doc         - Generate documentation"
	@echo "  audit       - Run security audit"
	@echo "  ci          - Run all CI checks"
	@echo "  setup       - Install development dependencies"
	@echo "  test-features - Test with different feature combinations"
	@echo "  examples    - Run examples"
	@echo "  msrv        - Check minimum supported Rust version"
	@echo "  help        - Show this help message"