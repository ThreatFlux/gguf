//! Async usage example for the gguf_rs library
//!
//! This example demonstrates how to read a GGUF file asynchronously.

#[cfg(feature = "async")]
use gguf::prelude::*;
#[cfg(feature = "async")]
use std::env;
#[cfg(feature = "async")]
use tokio;

#[cfg(feature = "async")]
#[tokio::main]
async fn main() -> Result<()> {
    // Get the GGUF file path from command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path_to_gguf_file>", args[0]);
        std::process::exit(1);
    }

    let file_path = &args[1];
    println!("Reading GGUF file asynchronously: {}", file_path);

    // Open and read the GGUF file asynchronously
    let file = tokio::fs::File::open(file_path).await.map_err(|e| GGUFError::Io(e))?;

    let gguf = GGUFFile::read_async(file).await?;

    // Display basic file information
    println!("\n=== GGUF File Information ===");
    println!("GGUF Version: {}", gguf.version());
    println!("Number of tensors: {}", gguf.tensors().len());
    println!("Number of metadata entries: {}", gguf.metadata().len());

    // Display some metadata
    println!("\n=== Sample Metadata ===");
    let mut count = 0;
    for (key, value) in gguf.metadata().iter() {
        println!("{}: {}", key, value);
        count += 1;
        if count >= 5 {
            if gguf.metadata().len() > 5 {
                println!("... and {} more entries", gguf.metadata().len() - 5);
            }
            break;
        }
    }

    // Display tensor summary
    println!("\n=== Tensor Summary ===");
    if gguf.tensors().is_empty() {
        println!("No tensors found");
    } else {
        // Group tensors by type
        let mut type_counts = std::collections::HashMap::new();
        let mut total_size = 0u64;

        for tensor in gguf.tensors() {
            *type_counts.entry(tensor.tensor_type()).or_insert(0) += 1;
            total_size += tensor.data().len() as u64;
        }

        println!("Total tensors: {}", gguf.tensors().len());
        println!("Total size: {} bytes ({:.2} MB)", total_size, total_size as f64 / 1_048_576.0);

        println!("\nTensors by type:");
        for (tensor_type, count) in type_counts {
            println!("  {}: {} tensors", tensor_type, count);
        }
    }

    println!("\nAsync reading completed successfully!");
    Ok(())
}

#[cfg(not(feature = "async"))]
fn main() {
    eprintln!("This example requires the 'async' feature to be enabled.");
    eprintln!("Run with: cargo run --example async_usage --features async");
    std::process::exit(1);
}

#[cfg(all(feature = "async", test))]
mod tests {
    use super::*;
    use tokio::io::Cursor;

    #[tokio::test]
    async fn test_async_example_with_minimal_gguf() {
        // Create minimal GGUF data for testing
        let mut data = Vec::new();
        data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF magic
        data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count
        data.extend_from_slice(&0u64.to_le_bytes()); // Metadata count

        let cursor = Cursor::new(data);
        let gguf = GGUFFile::read_async(cursor).await.unwrap();

        assert_eq!(gguf.version(), 3);
        assert_eq!(gguf.tensors().len(), 0);
        assert_eq!(gguf.metadata().len(), 0);
    }
}
