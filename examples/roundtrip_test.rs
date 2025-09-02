//! GGUF roundtrip test - create a file and read it back to verify everything works
//!
//! This example requires the `std` feature because it uses file I/O operations.

#[cfg(feature = "std")]
use gguf_rs::format::Metadata as FormatMetadata;
#[cfg(feature = "std")]
use gguf_rs::prelude::*;
#[cfg(feature = "std")]
use gguf_rs::reader::GGUFStreamReader;
#[cfg(feature = "std")]
use gguf_rs::tensor::{TensorData, TensorInfo, TensorShape, TensorType};
#[cfg(feature = "std")]
use gguf_rs::writer::GGUFStreamWriter;
#[cfg(feature = "std")]
use std::io::Cursor;

#[cfg(feature = "std")]
fn main() -> Result<()> {
    println!("🧪 GGUF Roundtrip Test");
    println!("======================");

    // Test 1: Create minimal GGUF in memory and read it back
    println!("\n📝 Test 1: Minimal GGUF roundtrip");
    test_minimal_roundtrip()?;

    // Test 2: Create GGUF with metadata and tensors
    println!("\n📝 Test 2: Full GGUF roundtrip");
    test_full_roundtrip()?;

    // Test 3: Test with file I/O
    println!("\n📝 Test 3: File I/O roundtrip");
    test_file_roundtrip()?;

    println!("\n✅ All tests passed! The GGUF library is working correctly.");
    Ok(())
}

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("This example requires the 'std' feature to be enabled.");
    eprintln!("Run with: cargo run --example roundtrip_test --features std");
    std::process::exit(1);
}

#[cfg(feature = "std")]
fn test_minimal_roundtrip() -> Result<()> {
    // Create minimal GGUF data
    let mut buffer = Vec::new();

    // Create empty metadata and tensor list
    let metadata = FormatMetadata::new();
    let tensors = Vec::<(TensorInfo, TensorData)>::new();

    // Write to buffer using stream writer
    let cursor = Cursor::new(&mut buffer);
    let mut writer = GGUFStreamWriter::new(cursor);

    let _result = writer.write_complete_stream(&metadata, &tensors)?;

    println!("  ✅ Created minimal GGUF: {} bytes", buffer.len());

    // Read it back
    let cursor = Cursor::new(buffer);
    let reader = GGUFStreamReader::new(cursor)?;

    println!("  ✅ Read back successfully");
    println!("     - Version: {}", reader.header().version);
    println!("     - Tensors: {}", reader.tensor_count());
    println!("     - Metadata: {}", reader.metadata().len());

    assert_eq!(reader.header().version, 3);
    assert_eq!(reader.tensor_count(), 0);
    assert_eq!(reader.metadata().len(), 0);

    Ok(())
}

#[cfg(feature = "std")]
fn test_full_roundtrip() -> Result<()> {
    // Create metadata
    let mut metadata = FormatMetadata::new();
    metadata.insert("test.name".to_string(), MetadataValue::String("Test Model".to_string()));
    metadata.insert("test.version".to_string(), MetadataValue::U32(42));
    metadata.insert("test.temperature".to_string(), MetadataValue::F32(0.8));

    // Create a simple tensor
    let shape = TensorShape::new(vec![2, 3])?;
    let data = TensorData::new_owned(vec![
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    ]);
    let tensor_info = TensorInfo::new("test.weights".to_string(), shape, TensorType::F32, 0);

    let tensors = vec![(tensor_info, data)];

    // Write to buffer
    let mut buffer = Vec::new();
    let cursor = Cursor::new(&mut buffer);
    let mut writer = GGUFStreamWriter::new(cursor);

    let _result = writer.write_complete_stream(&metadata, &tensors)?;

    println!("  ✅ Created full GGUF: {} bytes", buffer.len());

    // Read it back
    let cursor = Cursor::new(buffer);
    let mut reader = GGUFStreamReader::new(cursor)?;

    println!("  ✅ Read back successfully");
    println!("     - Version: {}", reader.header().version);
    println!("     - Tensors: {}", reader.tensor_count());
    println!("     - Metadata: {}", reader.metadata().len());

    // Verify metadata
    if let Some(MetadataValue::String(name)) = reader.metadata().get("test.name") {
        assert_eq!(name, "Test Model");
    } else {
        panic!("Expected string metadata value for test.name");
    }
    if let Some(MetadataValue::U32(version)) = reader.metadata().get("test.version") {
        assert_eq!(*version, 42);
    } else {
        panic!("Expected u32 metadata value for test.version");
    }
    if let Some(MetadataValue::F32(temp)) = reader.metadata().get("test.temperature") {
        assert_eq!(*temp, 0.8);
    } else {
        panic!("Expected f32 metadata value for test.temperature");
    }

    // Verify tensor info
    assert_eq!(reader.tensor_count(), 1);
    let tensor_info = &reader.tensor_infos()[0];
    assert_eq!(tensor_info.name(), "test.weights");
    assert_eq!(tensor_info.shape().dims(), &[2, 3]);
    assert_eq!(tensor_info.tensor_type(), TensorType::F32);

    // Read tensor data
    let tensor_data = reader.read_all_tensors()?;
    assert_eq!(tensor_data.len(), 1);
    let data = tensor_data.get("test.weights").unwrap();
    assert_eq!(data.len(), 24);

    Ok(())
}

#[cfg(feature = "std")]
fn test_file_roundtrip() -> Result<()> {
    let file_path = "test_roundtrip.gguf";

    // Create file
    {
        let mut metadata = FormatMetadata::new();
        metadata
            .insert("file.test".to_string(), MetadataValue::String("File I/O Test".to_string()));

        let shape = TensorShape::new(vec![4])?;
        let data = TensorData::new_owned(vec![0u8; 16]); // 4 F32 values
        let tensor_info = TensorInfo::new("file.tensor".to_string(), shape, TensorType::F32, 0);

        let tensors = vec![(tensor_info, data)];

        // Write to file using file writer
        let file = std::fs::File::create(file_path)?;
        let mut writer = GGUFFileWriter::new(file);

        let result = writer.write_complete_file(&metadata, &tensors)?;
        println!("  ✅ Created file: {} bytes written", result.total_bytes_written);
    }

    // Read it back
    {
        let file = std::fs::File::open(file_path)?;
        let reader = GGUFFileReader::new(file)?;

        println!("  ✅ Read file successfully");
        println!("     - Version: {}", reader.header().version);
        println!("     - Tensors: {}", reader.tensor_infos().len());
        println!("     - Metadata: {}", reader.metadata().len());

        // Verify
        assert_eq!(reader.metadata().get_string("file.test"), Some("File I/O Test"));
        assert_eq!(reader.tensor_infos().len(), 1);
        assert_eq!(reader.tensor_infos()[0].name(), "file.tensor");
    }

    // Clean up
    std::fs::remove_file(file_path).ok();

    Ok(())
}
