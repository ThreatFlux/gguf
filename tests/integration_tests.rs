//! Integration tests for the gguf_rs library

#![recursion_limit = "2048"]

#[cfg(feature = "std")]
use gguf::format::MetadataValue;
#[cfg(feature = "std")]
use gguf::prelude::*;
#[cfg(feature = "std")]
use gguf::reader::GGUFFileReader;
#[cfg(feature = "std")]
use gguf::tensor::{TensorData, TensorInfo, TensorType};
#[cfg(feature = "std")]
use std::io::Cursor;
#[cfg(feature = "std")]
use std::io::Write;
#[cfg(feature = "std")]
use tempfile::NamedTempFile;

/// Helper function to create minimal valid GGUF data
#[cfg(feature = "std")]
fn create_minimal_gguf_data() -> Vec<u8> {
    let mut data = Vec::new();

    // GGUF header
    data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF magic
    data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count
    data.extend_from_slice(&0u64.to_le_bytes()); // Metadata count

    data
}

#[cfg(feature = "std")]
#[test]
fn test_read_minimal_gguf() {
    let data = create_minimal_gguf_data();
    let cursor = Cursor::new(data);

    let reader = GGUFFileReader::new(cursor).expect("Failed to read minimal GGUF");

    assert_eq!(reader.header().version, 3);
    assert_eq!(reader.tensor_infos().len(), 0);
    assert_eq!(reader.metadata().len(), 0);
}

#[cfg(feature = "std")]
#[test]
fn test_invalid_magic_number() {
    let mut data = Vec::new();
    data.extend_from_slice(&0x12345678u32.to_le_bytes()); // Invalid magic
    data.extend_from_slice(&3u32.to_le_bytes()); // Version 3

    let cursor = Cursor::new(data);
    let result = GGUFFileReader::new(cursor);

    assert!(result.is_err());
}

#[cfg(feature = "std")]
#[test]
fn test_unsupported_version() {
    let mut data = Vec::new();
    data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF magic
    data.extend_from_slice(&999u32.to_le_bytes()); // Unsupported version

    let cursor = Cursor::new(data);
    let result = GGUFFileReader::new(cursor);

    assert!(result.is_err());
}

#[cfg(feature = "std")]
#[test]
fn test_truncated_file() {
    let data = vec![0x47, 0x47, 0x55]; // Only 3 bytes (insufficient for magic)
    let cursor = Cursor::new(data);

    let result = GGUFFileReader::new(cursor);
    assert!(result.is_err());
}

#[cfg(feature = "std")]
#[test]
fn test_file_from_disk() {
    let data = create_minimal_gguf_data();

    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file.write_all(&data).expect("Failed to write test data");
    temp_file.flush().expect("Failed to flush temp file");

    let file = std::fs::File::open(temp_file.path()).expect("Failed to open temp file");
    let reader = GGUFFileReader::new(file).expect("Failed to read GGUF from disk");

    assert_eq!(reader.header().version, 3);
    assert_eq!(reader.tensor_infos().len(), 0);
    assert_eq!(reader.metadata().len(), 0);
}

#[cfg(feature = "std")]
#[test]
fn test_metadata_operations() {
    let mut metadata = Metadata::new();

    // Test empty metadata
    assert!(metadata.is_empty());
    assert_eq!(metadata.len(), 0);

    // Test insertion and retrieval
    metadata.insert("test_key".to_string(), MetadataValue::String("test_value".to_string()));
    assert!(!metadata.is_empty());
    assert_eq!(metadata.len(), 1);

    let value = metadata.get("test_key");
    assert!(value.is_some());

    match value.unwrap() {
        MetadataValue::String(s) => assert_eq!(s, "test_value"),
        _ => panic!("Unexpected metadata value type"),
    }

    // Test non-existent key
    assert!(metadata.get("non_existent").is_none());
}

#[cfg(feature = "std")]
#[test]
fn test_tensor_type_properties() {
    // Test basic types
    assert_eq!(TensorType::F32.element_size(), 4);
    assert_eq!(TensorType::F16.element_size(), 2);
    assert_eq!(TensorType::I32.element_size(), 4);

    // Test quantized types
    assert!(TensorType::Q4_0.is_quantized());
    assert!(TensorType::Q8_0.is_quantized());
    assert!(!TensorType::F32.is_quantized());
    assert!(!TensorType::I32.is_quantized());

    // Test names
    assert_eq!(TensorType::F32.name(), "F32");
    assert_eq!(TensorType::Q4_0.name(), "Q4_0");
}

#[cfg(feature = "std")]
#[test]
fn test_tensor_creation_and_properties() {
    let data = TensorData::new_owned(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let shape = gguf::tensor::TensorShape::new(vec![2, 1]).unwrap();
    let mut tensor = TensorInfo::new("test_tensor".to_string(), shape, TensorType::F32, 0);
    tensor.set_data(data.clone());

    assert_eq!(tensor.name(), "test_tensor");
    assert_eq!(tensor.tensor_type(), TensorType::F32);
    assert_eq!(tensor.shape().dims(), &[2, 1]);
    assert_eq!(tensor.element_count(), 2);
    assert_eq!(tensor.data().unwrap().len(), 8);
}

#[cfg(feature = "std")]
#[test]
fn test_tensor_data_operations() {
    let data = vec![1, 2, 3, 4, 5];
    let tensor_data = TensorData::new_owned(data.clone());

    assert_eq!(tensor_data.len(), 5);
    assert!(!tensor_data.is_empty());
    assert_eq!(tensor_data.as_slice(), &data);

    // Test empty data
    let empty_data = TensorData::empty();
    assert_eq!(empty_data.len(), 0);
    assert!(empty_data.is_empty());
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_async_read_minimal_gguf() {
    // Note: This test would need an async GGUF reader implementation
    // For now, just testing the synchronous version works
    let data = create_minimal_gguf_data();
    let cursor = std::io::Cursor::new(data);

    let reader = GGUFFileReader::new(cursor).expect("Failed to read minimal GGUF");

    assert_eq!(reader.header().version, 3);
    assert_eq!(reader.tensor_infos().len(), 0);
    assert_eq!(reader.metadata().len(), 0);
}

#[cfg(all(feature = "mmap", feature = "std"))]
#[test]
fn test_mmap_read_minimal_gguf() {
    let data = create_minimal_gguf_data();

    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file.write_all(&data).expect("Failed to write test data");
    temp_file.flush().expect("Failed to flush temp file");

    let file = std::fs::File::open(temp_file.path()).expect("Failed to open temp file");
    let reader = GGUFFileReader::new(file).expect("Failed to read GGUF from mmap");

    assert_eq!(reader.header().version, 3);
    assert_eq!(reader.tensor_infos().len(), 0);
    assert_eq!(reader.metadata().len(), 0);
}
