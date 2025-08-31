//! Integration tests for the gguf_rs library

use gguf::prelude::*;
use std::io::Cursor;
use std::io::Write;
use tempfile::NamedTempFile;

/// Helper function to create minimal valid GGUF data
fn create_minimal_gguf_data() -> Vec<u8> {
    let mut data = Vec::new();

    // GGUF header
    data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF magic
    data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count
    data.extend_from_slice(&0u64.to_le_bytes()); // Metadata count

    data
}

#[test]
fn test_read_minimal_gguf() {
    let data = create_minimal_gguf_data();
    let cursor = Cursor::new(data);

    let gguf = GGUFFile::read(cursor).expect("Failed to read minimal GGUF");

    assert_eq!(gguf.version(), 3);
    assert_eq!(gguf.tensors().len(), 0);
    assert_eq!(gguf.metadata().len(), 0);
}

#[test]
fn test_invalid_magic_number() {
    let mut data = Vec::new();
    data.extend_from_slice(&0x12345678u32.to_le_bytes()); // Invalid magic
    data.extend_from_slice(&3u32.to_le_bytes()); // Version 3

    let cursor = Cursor::new(data);
    let result = GGUFFile::read(cursor);

    assert!(matches!(result, Err(GGUFError::InvalidMagic { .. })));
}

#[test]
fn test_unsupported_version() {
    let mut data = Vec::new();
    data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF magic
    data.extend_from_slice(&999u32.to_le_bytes()); // Unsupported version

    let cursor = Cursor::new(data);
    let result = GGUFFile::read(cursor);

    assert!(matches!(result, Err(GGUFError::UnsupportedVersion(999))));
}

#[test]
fn test_truncated_file() {
    let data = vec![0x47, 0x47, 0x55]; // Only 3 bytes (insufficient for magic)
    let cursor = Cursor::new(data);

    let result = GGUFFile::read(cursor);
    assert!(result.is_err());
}

#[test]
fn test_file_from_disk() {
    let data = create_minimal_gguf_data();

    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file.write_all(&data).expect("Failed to write test data");
    temp_file.flush().expect("Failed to flush temp file");

    let file = std::fs::File::open(temp_file.path()).expect("Failed to open temp file");
    let gguf = GGUFFile::read(file).expect("Failed to read GGUF from disk");

    assert_eq!(gguf.version(), 3);
    assert_eq!(gguf.tensors().len(), 0);
    assert_eq!(gguf.metadata().len(), 0);
}

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

#[test]
fn test_tensor_type_properties() {
    // Test basic types
    assert_eq!(TensorType::F32.size_in_bytes(), 4);
    assert_eq!(TensorType::F16.size_in_bytes(), 2);
    assert_eq!(TensorType::I32.size_in_bytes(), 4);

    // Test quantized types
    assert!(TensorType::Q4_0.is_quantized());
    assert!(TensorType::Q8_0.is_quantized());
    assert!(!TensorType::F32.is_quantized());
    assert!(!TensorType::I32.is_quantized());

    // Test names
    assert_eq!(TensorType::F32.name(), "F32");
    assert_eq!(TensorType::Q4_0.name(), "Q4_0");
}

#[test]
fn test_tensor_creation_and_properties() {
    let data = TensorData::Bytes(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let tensor = Tensor::new("test_tensor".to_string(), TensorType::F32, vec![2, 1], data);

    assert_eq!(tensor.name(), "test_tensor");
    assert_eq!(tensor.tensor_type(), TensorType::F32);
    assert_eq!(tensor.shape(), &[2, 1]);
    assert_eq!(tensor.element_count(), 2);
    assert_eq!(tensor.element_size(), 4);
    assert_eq!(tensor.data().len(), 8);
}

#[test]
fn test_tensor_data_operations() {
    let data = vec![1, 2, 3, 4, 5];
    let tensor_data = TensorData::Bytes(data.clone());

    assert_eq!(tensor_data.len(), 5);
    assert!(!tensor_data.is_empty());
    assert_eq!(tensor_data.as_slice(), &data);

    // Test empty data
    let empty_data = TensorData::Bytes(Vec::new());
    assert_eq!(empty_data.len(), 0);
    assert!(empty_data.is_empty());
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_async_read_minimal_gguf() {
    let data = create_minimal_gguf_data();
    let cursor = tokio::io::Cursor::new(data);

    let gguf = GGUFFile::read_async(cursor).await.expect("Failed to read minimal GGUF async");

    assert_eq!(gguf.version(), 3);
    assert_eq!(gguf.tensors().len(), 0);
    assert_eq!(gguf.metadata().len(), 0);
}

#[cfg(feature = "mmap")]
#[test]
fn test_mmap_read_minimal_gguf() {
    let data = create_minimal_gguf_data();

    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file.write_all(&data).expect("Failed to write test data");
    temp_file.flush().expect("Failed to flush temp file");

    let gguf = GGUFFile::mmap(temp_file.path()).expect("Failed to mmap GGUF");

    assert_eq!(gguf.version(), 3);
    assert_eq!(gguf.tensors().len(), 0);
    assert_eq!(gguf.metadata().len(), 0);
}
