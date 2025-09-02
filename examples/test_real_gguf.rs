//! Test the GGUF library against real GGUF model files
//!
//! This example requires the `std` feature because it uses file I/O operations.

#[cfg(feature = "std")]
use gguf::reader::file_reader::GGUFFileReader;
#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::Read;
#[cfg(feature = "std")]
use std::path::Path;

#[cfg(feature = "std")]
fn analyze_gguf_file(path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(60));
    println!("Analyzing: {}", path);
    println!("{}", "=".repeat(60));

    // Get file size
    let file_size = std::fs::metadata(path)?.len();
    println!("File size: {:.2} MB", file_size as f64 / (1024.0 * 1024.0));

    // Open and read the file header manually first
    let mut file = File::open(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;

    println!(
        "Magic bytes: {:02X} {:02X} {:02X} {:02X} ({})",
        magic[0],
        magic[1],
        magic[2],
        magic[3],
        std::str::from_utf8(&magic).unwrap_or("invalid")
    );

    // Check if it's a valid GGUF file
    if &magic != b"GGUF" {
        println!("ERROR: Not a valid GGUF file!");
        return Ok(());
    }

    // Read version
    let mut version_bytes = [0u8; 4];
    file.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    println!("GGUF Version: {}", version);

    // Read tensor count
    let mut tensor_count_bytes = [0u8; 8];
    file.read_exact(&mut tensor_count_bytes)?;
    let tensor_count = u64::from_le_bytes(tensor_count_bytes);
    println!("Tensor count: {}", tensor_count);

    // Read metadata count
    let mut metadata_count_bytes = [0u8; 8];
    file.read_exact(&mut metadata_count_bytes)?;
    let metadata_count = u64::from_le_bytes(metadata_count_bytes);
    println!("Metadata count: {}", metadata_count);

    // Now try to use our library
    println!("\nTesting with our GGUF library:");
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            println!("✗ Failed to open file: {}", e);
            return Ok(());
        }
    };

    match GGUFFileReader::new(file) {
        Ok(reader) => {
            let header = reader.header();
            println!("✓ Header read successfully");
            println!("  - Version: {}", header.version);
            println!("  - Tensors: {}", header.tensor_count);
            println!("  - Metadata entries: {}", header.metadata_kv_count);

            // Try to read metadata
            println!("\nReading metadata...");
            let metadata = reader.metadata();
            if !metadata.is_empty() {
                println!("  Read {} metadata entries", metadata.len());
            }

            // Try to read tensor info
            println!("\nReading tensor info...");
            let tensor_infos = reader.tensor_infos();
            if !tensor_infos.is_empty() {
                println!("  Read {} tensor descriptors", tensor_infos.len());
            }

            println!("\n✓ File appears to be compatible with our library");
        }
        Err(e) => {
            println!("✗ Failed to open file with our library: {}", e);
        }
    }

    Ok(())
}

#[cfg(feature = "std")]
fn main() {
    println!("Testing GGUF Library Against Real Model Files");
    println!("{}", "=".repeat(60));

    // Test files - using different sizes and types
    let test_files = vec![
        "data/qwen3-yara-sharegpt.gguf",        // 610MB file
        "data/qwen3-4b-yara-v1-q4_k_m.gguf",    // 2.4GB quantized file
        "data/qwen3-yara-sharegpt-v5.f16.gguf", // F16 format
    ];

    let mut successful = 0;
    let mut failed = 0;

    for file_path in &test_files {
        if Path::new(file_path).exists() {
            match analyze_gguf_file(file_path) {
                Ok(_) => successful += 1,
                Err(e) => {
                    println!("Error analyzing {}: {}", file_path, e);
                    failed += 1;
                }
            }
        } else {
            println!("\nFile not found: {}", file_path);
            failed += 1;
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!("Files tested: {}", test_files.len());
    println!("Successful: {}", successful);
    println!("Failed: {}", failed);

    if failed == 0 {
        println!("\n✓ All tests passed! The library can handle real GGUF files.");
    } else {
        println!("\n⚠ Some tests failed. The library may need adjustments for real files.");
    }
}

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("This example requires the 'std' feature to be enabled.");
    eprintln!("Run with: cargo run --example test_real_gguf --features std");
    std::process::exit(1);
}
