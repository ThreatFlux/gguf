//! Basic usage example for the gguf_rs library
//!
//! This example demonstrates how to read a GGUF file and inspect its contents.

use gguf::prelude::*;
use std::env;

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
    let file = std::fs::File::open(file_path).map_err(|e| GGUFError::Io(e))?;

    let gguf = GGUFFile::read(file)?;

    // Display basic file information
    println!("\n=== GGUF File Information ===");
    println!("GGUF Version: {}", gguf.version());
    println!("Number of tensors: {}", gguf.tensors().len());
    println!("Number of metadata entries: {}", gguf.metadata().len());

    // Display metadata
    println!("\n=== Metadata ===");
    if gguf.metadata().is_empty() {
        println!("No metadata found");
    } else {
        for (key, value) in gguf.metadata().iter() {
            println!("{}: {}", key, value);
        }
    }

    // Display tensor information
    println!("\n=== Tensors ===");
    if gguf.tensors().is_empty() {
        println!("No tensors found");
    } else {
        for (i, tensor) in gguf.tensors().iter().enumerate() {
            println!("Tensor {}: {}", i, tensor.name());
            println!("  Type: {}", tensor.tensor_type());
            println!("  Shape: {:?}", tensor.shape());
            println!("  Elements: {}", tensor.element_count());
            println!("  Size: {} bytes", tensor.data().len());

            if i >= 10 {
                println!("  ... and {} more tensors", gguf.tensors().len() - 10);
                break;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
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
        let gguf = GGUFFile::read(cursor).unwrap();

        assert_eq!(gguf.version(), 3);
        assert_eq!(gguf.tensors().len(), 0);
        assert_eq!(gguf.metadata().len(), 0);
    }
}
