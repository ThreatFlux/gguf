//! Test reading real GGUF files

use gguf::prelude::*;
use gguf::format::header::GGUFHeader;
use std::fs::File;
use std::io::BufReader;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Testing GGUF library with real files");
    println!("=====================================\n");
    
    // Test file
    let test_file = "data/qwen3-yara-sharegpt.gguf";
    
    println!("Reading: {}", test_file);
    
    // Open file
    let file = File::open(test_file)?;
    let mut reader = BufReader::new(file);
    
    // Read header
    let header = GGUFHeader::read_from(&mut reader).map_err(|e| format!("Failed to read header: {:?}", e))?;
    
    println!("✓ Successfully read GGUF header!");
    println!("  Magic: 0x{:08X}", header.magic);
    println!("  Version: {}", header.version);
    println!("  Tensor count: {}", header.tensor_count);
    println!("  Metadata count: {}", header.metadata_kv_count);
    
    println!("\n✅ GGUF library successfully read a real GGUF file!");
    
    Ok(())
}