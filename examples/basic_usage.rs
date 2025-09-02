//! Basic usage example for the gguf_rs library
//!
//! This example demonstrates how to read a GGUF file and inspect its contents.
//!
//! This example requires the `std` feature because it uses file I/O operations.

#[cfg(feature = "std")]
use gguf::prelude::*;
#[cfg(feature = "std")]
use std::env;

#[cfg(feature = "std")]
fn main() -> Result<()> {
    // Get the GGUF file path from command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path_to_gguf_file>", args[0]);
        std::process::exit(1);
    }

    let file_path = &args[1];
    println!("Reading GGUF file: {}", file_path);

    // Open and read the GGUF file
    let file = std::fs::File::open(file_path)?;

    let reader = GGUFFileReader::new(file)?;

    // Display basic file information
    println!("\n=== GGUF File Information ===");
    println!("GGUF Version: {}", reader.header().version);
    println!("Number of tensors: {}", reader.tensor_infos().len());
    println!("Number of metadata entries: {}", reader.metadata().len());

    // Display metadata
    println!("\n=== Metadata ===");
    if reader.metadata().is_empty() {
        println!("No metadata found");
    } else {
        for (key, value) in reader.metadata().iter() {
            println!("{}: {}", key, value);
        }
    }

    // Display tensor information
    println!("\n=== Tensors ===");
    if reader.tensor_infos().is_empty() {
        println!("No tensors found");
    } else {
        for (i, tensor) in reader.tensor_infos().iter().enumerate() {
            println!("Tensor {}: {}", i, tensor.name());
            println!("  Type: {}", tensor.tensor_type());
            println!("  Shape: {:?}", tensor.shape().dims());
            println!("  Elements: {}", tensor.element_count());
            println!("  Expected size: {} bytes", tensor.expected_data_size());

            if i >= 10 {
                println!("  ... and {} more tensors", reader.tensor_infos().len() - 10);
                break;
            }
        }
    }

    Ok(())
}

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("This example requires the 'std' feature to be enabled.");
    eprintln!("Run with: cargo run --example basic_usage --features std");
    std::process::exit(1);
}

#[cfg(all(feature = "std", test))]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_example_with_minimal_gguf() {
        // Create minimal GGUF data for testing
        let mut data = Vec::new();
        data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF magic
        data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count
        data.extend_from_slice(&0u64.to_le_bytes()); // Metadata count

        let cursor = Cursor::new(data);
        let reader = GGUFFileReader::new(cursor).unwrap();

        assert_eq!(reader.header().version, 3);
        assert_eq!(reader.tensor_infos().len(), 0);
        assert_eq!(reader.metadata().len(), 0);
    }
}
