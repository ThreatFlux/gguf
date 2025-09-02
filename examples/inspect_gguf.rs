//! Comprehensive GGUF file inspection tool
//!
//! This example demonstrates how to read and analyze GGUF files using the gguf_rs library.
//!
//! This example requires the `std` feature because it uses file I/O operations.

#[cfg(feature = "std")]
use gguf_rs::prelude::*;
#[cfg(feature = "std")]
use std::env;

#[cfg(feature = "std")]
fn main() -> Result<()> {
    // Get the GGUF file path from command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path_to_gguf_file>", args[0]);
        eprintln!("Example: {} model.gguf", args[0]);
        std::process::exit(1);
    }

    let file_path = &args[1];
    println!("🔍 Inspecting GGUF file: {}", file_path);

    // Check if file exists
    if !std::path::Path::new(file_path).exists() {
        eprintln!("❌ Error: File '{}' does not exist", file_path);
        std::process::exit(1);
    }

    // Get file size
    let file_size = std::fs::metadata(file_path)?.len();
    println!("📁 File size: {} bytes ({:.2} MB)", file_size, file_size as f64 / 1_048_576.0);

    // Open and read the GGUF file
    let file = std::fs::File::open(file_path).map_err(GGUFError::Io)?;
    let reader = GGUFFileReader::new(file)?;

    // Display basic file information
    println!("\n=== 📋 GGUF File Header ===");
    println!("GGUF Version: {}", reader.header().version);
    println!("Magic Number: 0x{:08X}", reader.header().magic);
    println!("Number of tensors: {}", reader.header().tensor_count);
    println!("Number of metadata entries: {}", reader.header().metadata_kv_count);

    // Display metadata
    println!("\n=== 🏷️  Metadata ({} entries) ===", reader.metadata().len());
    if reader.metadata().is_empty() {
        println!("No metadata found");
    } else {
        let mut metadata_items: Vec<_> = reader.metadata().iter().collect();
        metadata_items.sort_by_key(|(key, _)| *key);

        for (key, value) in metadata_items {
            match value {
                MetadataValue::String(s) => println!("  📝 {}: \"{}\"", key, s),
                MetadataValue::U32(n) => println!("  🔢 {}: {}", key, n),
                MetadataValue::U64(n) => println!("  🔢 {}: {}", key, n),
                MetadataValue::F32(f) => println!("  🔢 {}: {:.6}", key, f),
                MetadataValue::Bool(b) => println!("  ✅ {}: {}", key, b),
                _ => println!("  ❓ {}: {:?}", key, value),
            }
        }
    }

    // Display tensor information
    println!("\n=== 🧮 Tensors ({} tensors) ===", reader.tensor_infos().len());
    if reader.tensor_infos().is_empty() {
        println!("No tensors found");
    } else {
        let mut total_params = 0u64;
        let mut total_size = 0u64;

        for (i, tensor) in reader.tensor_infos().iter().enumerate() {
            let elements = tensor.element_count();
            let size_bytes = tensor.expected_data_size();
            total_params += elements;
            total_size += size_bytes;

            println!("  Tensor {}: 📊 {}", i + 1, tensor.name());
            println!("    Type: {} 🏷️", tensor.tensor_type().name());
            println!("    Shape: {} 📐", tensor.shape());
            println!("    Elements: {} 🔢", format_number(elements));
            println!("    Size: {} bytes ({}) 💾", size_bytes, format_bytes(size_bytes));
            println!("    Offset: {} 📍", tensor.data_offset());

            if i >= 19 {
                println!("    ... and {} more tensors 📦", reader.tensor_infos().len() - 20);
                break;
            }
            println!();
        }

        println!("📈 Summary:");
        println!("  Total parameters: {} 🧮", format_number(total_params));
        println!("  Total tensor data: {} bytes ({}) 💾", total_size, format_bytes(total_size));
    }

    // Analyze tensor types
    analyze_tensor_types(reader.tensor_infos());

    println!("\n✅ GGUF file inspection complete!");

    Ok(())
}

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("This example requires the 'std' feature to be enabled.");
    eprintln!("Run with: cargo run --example inspect_gguf --features std");
    std::process::exit(1);
}

#[cfg(feature = "std")]
fn analyze_tensor_types(tensors: &[gguf_rs::tensor::TensorInfo]) {
    if tensors.is_empty() {
        return;
    }

    println!("\n=== 🔬 Tensor Type Analysis ===");

    let mut type_counts = std::collections::HashMap::new();
    let mut type_sizes = std::collections::HashMap::new();

    for tensor in tensors {
        let tensor_type = tensor.tensor_type();
        *type_counts.entry(tensor_type).or_insert(0) += 1;
        *type_sizes.entry(tensor_type).or_insert(0u64) += tensor.expected_data_size();
    }

    let mut types: Vec<_> = type_counts.keys().collect();
    types.sort_by_key(|t| t.name());

    for tensor_type in types {
        let count = type_counts[tensor_type];
        let size = type_sizes[tensor_type];
        println!(
            "  {} 🏷️: {} tensors, {} ({}) 📊",
            tensor_type.name(),
            count,
            format_bytes(size),
            size
        );
    }
}

#[cfg(feature = "std")]
fn format_number(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

#[cfg(feature = "std")]
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1_024 {
        format!("{:.2} KB", bytes as f64 / 1_024.0)
    } else {
        format!("{} B", bytes)
    }
}
