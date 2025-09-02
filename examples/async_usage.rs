//! Async usage example for the gguf_rs library
//!
//! This example demonstrates how to read a GGUF file asynchronously.

#[cfg(feature = "async")]
use gguf_rs_lib::prelude::*;
#[cfg(feature = "async")]
use std::env;

#[cfg(feature = "async")]
#[tokio::main(flavor = "multi_thread")]
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
    let file = tokio::fs::File::open(file_path).await.map_err(GGUFError::Io)?;

    // Convert tokio::fs::File to std::fs::File for now
    // TODO: Implement actual async GGUF reading
    let std_file = file.into_std().await;
    let reader = gguf_rs_lib::reader::file_reader::GGUFFileReader::new(std_file)?;

    // Display basic file information
    println!("\n=== GGUF File Information ===");
    println!("GGUF Version: {}", reader.header().version);
    println!("Number of tensors: {}", reader.tensor_infos().len());
    println!("Number of metadata entries: {}", reader.metadata().len());

    // Display some metadata
    println!("\n=== Sample Metadata ===");
    let mut count = 0;
    for (key, value) in reader.metadata().iter() {
        println!("{}: {}", key, value);
        count += 1;
        if count >= 5 {
            if reader.metadata().len() > 5 {
                println!("... and {} more entries", reader.metadata().len() - 5);
            }
            break;
        }
    }

    // Display tensor summary
    println!("\n=== Tensor Summary ===");
    if reader.tensor_infos().is_empty() {
        println!("No tensors found");
    } else {
        // Group tensors by type
        let mut type_counts = std::collections::HashMap::new();
        let mut total_size = 0u64;

        for tensor in reader.tensor_infos() {
            *type_counts.entry(tensor.tensor_type()).or_insert(0) += 1;
            total_size += tensor.expected_data_size();
        }

        println!("Total tensors: {}", reader.tensor_infos().len());
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
    use std::io::Cursor;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_async_example_with_minimal_gguf() {
        // Create minimal GGUF data for testing
        let mut data = Vec::new();
        data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF magic
        data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count
        data.extend_from_slice(&0u64.to_le_bytes()); // Metadata count

        // For now, test the sync reader since async isn't fully implemented
        let cursor = Cursor::new(data);
        let reader = gguf_rs_lib::reader::file_reader::GGUFFileReader::new(cursor).unwrap();

        assert_eq!(reader.header().version, 3);
        assert_eq!(reader.header().tensor_count, 0);
        assert_eq!(reader.header().metadata_kv_count, 0);
    }
}
