#!/bin/bash
set -euo pipefail

# Development environment setup script for gguf_rs

echo "🦀 Setting up gguf_rs development environment..."

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "❌ Rust is not installed. Please install Rust from https://rustup.rs/"
    exit 1
fi

echo "✅ Rust is installed: $(rustc --version)"

# Install required toolchain components
echo "🔧 Installing required Rust components..."
rustup component add rustfmt clippy

# Install additional tools for development
echo "📦 Installing additional development tools..."

# cargo-audit for security auditing
if ! command -v cargo-audit &> /dev/null; then
    echo "Installing cargo-audit..."
    cargo install cargo-audit
fi

# cargo-tarpaulin for code coverage (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]] && ! command -v cargo-tarpaulin &> /dev/null; then
    echo "Installing cargo-tarpaulin..."
    cargo install cargo-tarpaulin
fi

# cargo-watch for automatic rebuilding during development
if ! command -v cargo-watch &> /dev/null; then
    echo "Installing cargo-watch..."
    cargo install cargo-watch
fi

# Pre-commit hooks setup
echo "🪝 Setting up Git hooks..."
mkdir -p .git/hooks

cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
set -e

echo "Running pre-commit checks..."

# Check formatting
cargo fmt --check || {
    echo "❌ Code is not formatted. Run 'cargo fmt' to fix."
    exit 1
}

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings || {
    echo "❌ Clippy found issues. Fix them before committing."
    exit 1
}

# Run tests
cargo test --all-features || {
    echo "❌ Tests failed. Fix them before committing."
    exit 1
}

echo "✅ All pre-commit checks passed!"
EOF

chmod +x .git/hooks/pre-commit

# Verify installation
echo "🧪 Verifying installation..."
cargo check --all-features
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features

echo "🎉 Development environment setup complete!"
echo ""
echo "Available commands:"
echo "  cargo build               - Build the library"
echo "  cargo test                - Run tests"
echo "  cargo test --all-features - Run tests with all features"
echo "  cargo bench               - Run benchmarks"
echo "  cargo fmt                 - Format code"
echo "  cargo clippy              - Run linter"
echo "  cargo doc --open          - Generate and open documentation"
echo "  cargo watch -x test       - Auto-run tests on file changes"
echo "  cargo audit               - Security audit dependencies"
echo ""
echo "CLI commands:"
echo "  cargo run -p gguf-cli -- --help    - Show CLI help"
echo "  cargo build -p gguf-cli --release  - Build CLI in release mode"
echo ""
echo "Happy coding! 🚀"