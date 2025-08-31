# Contributing to gguf_rs

Thank you for your interest in contributing to `gguf_rs`! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Rust 1.70.0 or later
- Git
- A GitHub account

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gguf_rs.git
   cd gguf_rs
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/ThreatFlux/gguf_rs.git
   ```
4. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Building and Testing

```bash
# Build the library
cargo build

# Run tests
cargo test

# Run tests with all features
cargo test --all-features

# Build and test the CLI
cargo build -p gguf-cli
cargo test -p gguf-cli

# Run benchmarks
cargo bench

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings

# Generate documentation
cargo doc --all-features --open
```

## Code Style and Quality

### Formatting

We use `rustfmt` to maintain consistent code formatting. The configuration is in `rustfmt.toml`. Please run `cargo fmt` before submitting your changes.

### Linting

We use Clippy with strict settings to catch common issues. Run `cargo clippy` and address any warnings or errors.

### Documentation

- All public APIs must be documented with doc comments
- Use `//!` for module-level documentation
- Use `///` for item documentation
- Include examples in doc comments where appropriate
- Run `cargo doc` to verify documentation builds correctly

### Testing

- Write unit tests for new functionality
- Add integration tests for end-to-end functionality
- Ensure all tests pass with `cargo test --all-features`
- Add benchmarks for performance-critical code
- Test with different feature combinations

## Contributing Guidelines

### Issues

- Search existing issues before creating a new one
- Use the provided issue templates
- Provide clear reproduction steps for bugs
- Include relevant system information

### Pull Requests

1. Create a focused pull request that addresses a single concern
2. Write a clear title and description
3. Include tests for new functionality
4. Update documentation as needed
5. Ensure CI passes
6. Request review from maintainers

### Commit Messages

Use conventional commit format:

- `feat: add support for GGUF v4 format`
- `fix: handle malformed tensor data gracefully`
- `docs: update README with new examples`
- `test: add integration tests for async functionality`
- `refactor: simplify metadata parsing logic`

### Code Review Process

1. All changes must be reviewed by at least one maintainer
2. Address feedback promptly and thoroughly
3. Keep discussions constructive and focused
4. Squash commits before merging if requested

## Feature Development

### Adding New Features

1. Discuss the feature in an issue first
2. Start with a minimal implementation
3. Add comprehensive tests
4. Update documentation
5. Consider backward compatibility
6. Add feature flags for optional functionality

### API Design Principles

- **Safety**: Prefer safe Rust, document any unsafe usage
- **Performance**: Optimize for common use cases, provide alternatives for edge cases
- **Ergonomics**: Design intuitive APIs that prevent common mistakes
- **Compatibility**: Maintain backward compatibility when possible
- **Extensibility**: Design for future enhancements

### Feature Flags

Use feature flags appropriately:

- `std`: Standard library support (default)
- `async`: Async I/O support
- `mmap`: Memory mapping support
- `serde`: Serialization support

Add new features behind feature flags when they:
- Add significant dependencies
- Are platform-specific
- Are experimental or unstable

## Testing Strategy

### Unit Tests

- Test individual functions and methods
- Use property-based testing for complex logic
- Mock external dependencies
- Test error conditions

### Integration Tests

- Test end-to-end functionality
- Use real GGUF files when possible
- Test different feature combinations
- Verify CLI functionality

### Benchmarks

- Benchmark performance-critical paths
- Include memory usage measurements
- Compare against baseline performance
- Document performance characteristics

## Documentation

### Code Documentation

```rust
/// Reads a GGUF file from a reader.
///
/// This function parses the GGUF header, metadata, and tensor information
/// from the provided reader. It performs validation to ensure the file
/// follows the GGUF specification.
///
/// # Arguments
///
/// * `reader` - A type implementing `Read` trait
///
/// # Returns
///
/// * `Ok(GGUFFile)` - Successfully parsed GGUF file
/// * `Err(GGUFError)` - Parsing error occurred
///
/// # Examples
///
/// ```rust
/// use gguf::GGUFFile;
/// use std::fs::File;
///
/// let file = File::open("model.gguf")?;
/// let gguf = GGUFFile::read(file)?;
/// println!("Loaded {} tensors", gguf.tensors().len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// This function will return an error if:
/// - The file has an invalid magic number
/// - The GGUF version is unsupported
/// - The file is truncated or malformed
pub fn read<R: Read>(reader: R) -> Result<Self> {
    // Implementation...
}
```

### README and Guides

- Keep README up-to-date with latest features
- Provide working examples
- Document common use cases
- Include performance characteristics
- Link to detailed documentation

## Release Process

### Versioning

We follow Semantic Versioning (SemVer):

- **Major**: Breaking API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

### Changelog

- Maintain CHANGELOG.md
- Document all user-facing changes
- Categorize changes (Added, Changed, Deprecated, Removed, Fixed, Security)
- Include migration notes for breaking changes

## Getting Help

- Join discussions in GitHub Issues
- Ask questions in GitHub Discussions
- Check existing documentation and examples
- Review the code for similar patterns

## Recognition

Contributors are recognized in:
- CONTRIBUTORS file
- Release notes
- Git commit history

Thank you for contributing to `gguf_rs`!