# gguf_rs

[![Crates.io](https://img.shields.io/crates/v/gguf.svg)](https://crates.io/crates/gguf)
[![Documentation](https://docs.rs/gguf/badge.svg)](https://docs.rs/gguf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ThreatFlux/gguf_rs/workflows/CI/badge.svg)](https://github.com/ThreatFlux/gguf_rs/actions)

A Rust library for reading and writing GGUF (GGML Universal Format) files, designed for machine learning model storage and manipulation.

## Overview

GGUF (GGML Universal Format) is a binary format for storing machine learning models, particularly those used with the GGML library. This crate provides a safe, efficient, and ergonomic interface for working with GGUF files in Rust.

## Features

- 🚀 **Fast and Memory Efficient**: Zero-copy parsing where possible with optional memory mapping support
- 🔒 **Type Safe**: Strongly typed API that prevents common errors when working with GGUF files
- 📦 **Serde Integration**: Built-in serialization and deserialization support
- 🎯 **No Unsafe Code**: Pure safe Rust implementation (by default)
- 🔄 **Async Support**: Optional async I/O support with Tokio
- 🛠️ **CLI Tool**: Command-line utility for inspecting and manipulating GGUF files
- 📚 **Well Documented**: Comprehensive documentation with examples
- 🧪 **Thoroughly Tested**: Extensive test suite including property-based tests

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
gguf = "0.2.0"
```

### Basic Usage

```rust
use gguf::{GGUFFile, GGUFError};
use std::fs::File;

fn main() -> Result<(), GGUFError> {
    // Read a GGUF file
    let file = File::open("model.gguf")?;
    let gguf = GGUFFile::read(file)?;
    
    // Access metadata
    println!("Model name: {}", gguf.metadata().get("general.name").unwrap());
    println!("Number of tensors: {}", gguf.tensors().len());
    
    // Iterate over tensors
    for tensor in gguf.tensors() {
        println!("Tensor: {} ({:?})", tensor.name(), tensor.shape());
    }
    
    Ok(())
}
```

### Async Usage (with `async` feature)

```rust
use gguf::{GGUFFile, GGUFError};
use tokio::fs::File;

#[tokio::main]
async fn main() -> Result<(), GGUFError> {
    let file = File::open("model.gguf").await?;
    let gguf = GGUFFile::read_async(file).await?;
    
    // Work with the GGUF file asynchronously
    println!("Loaded {} tensors", gguf.tensors().len());
    
    Ok(())
}
```

### Memory Mapping (with `mmap` feature)

```rust
use gguf::{GGUFFile, GGUFError};

fn main() -> Result<(), GGUFError> {
    // Memory map a large GGUF file for efficient access
    let gguf = GGUFFile::mmap("large_model.gguf")?;
    
    // Access data without loading entire file into memory
    for tensor in gguf.tensors() {
        let data = tensor.data(); // Zero-copy access to tensor data
        // Process tensor data...
    }
    
    Ok(())
}
```

## CLI Tool

The `gguf-cli` tool provides command-line access to GGUF functionality:

```bash
# Install the CLI tool
cargo install gguf --features=cli

# Inspect a GGUF file
gguf-cli info model.gguf

# List all tensors
gguf-cli tensors model.gguf

# Extract metadata
gguf-cli metadata model.gguf --format json

# Validate file integrity
gguf-cli validate model.gguf
```

## Feature Flags

- `std` (default): Standard library support
- `async`: Async I/O support with Tokio
- `mmap`: Memory mapping support for large files
- `cli`: Build the command-line tool

## Performance

This library is designed for performance:

- Zero-copy parsing where possible
- Optional memory mapping for large files
- Efficient tensor data access
- Minimal allocations during parsing

Benchmarks show that `gguf_rs` can parse large GGUF files significantly faster than equivalent Python implementations.

## Safety

This crate uses only safe Rust by default. The optional `mmap` feature uses memory mapping, which involves some inherent platform-specific risks, but the API remains safe to use.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ThreatFlux/gguf_rs.git
cd gguf_rs

# Run tests
cargo test

# Run benchmarks
cargo bench

# Check formatting and linting
cargo fmt --check
cargo clippy -- -D warnings
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Claude Code
- Wyatt Roersma

## Acknowledgments

- The GGML project for the GGUF format specification
- The Rust community for excellent crates that make this library possible

## Related Projects

- [ggml](https://github.com/ggerganov/ggml) - The original GGML library
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ implementation using GGUF