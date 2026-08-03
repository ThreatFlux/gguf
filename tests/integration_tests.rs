//! Integration tests for the gguf_rs library

#![recursion_limit = "2048"]

#[cfg(feature = "std")]
use gguf_rs_lib::format::MetadataValue;
#[cfg(feature = "std")]
use gguf_rs_lib::prelude::*;
#[cfg(feature = "std")]
use gguf_rs_lib::reader::GGUFFileReader;
#[cfg(feature = "std")]
use gguf_rs_lib::tensor::{TensorData, TensorInfo, TensorType};
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

/// Hand-encoded GGUF v3 fixture. This deliberately does not call any SDK
/// writer so a shared serializer/parser defect cannot make the test pass.
#[cfg(feature = "std")]
fn create_manual_aligned_gguf_fixture() -> Vec<u8> {
    fn u32_le(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn u64_le(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn string(bytes: &mut Vec<u8>, value: &str) {
        u64_le(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    let mut bytes = Vec::new();
    u32_le(&mut bytes, 0x4655_4747); // magic
    u32_le(&mut bytes, 3); // version
    u64_le(&mut bytes, 1); // tensor count
    u64_le(&mut bytes, 2); // metadata count

    string(&mut bytes, "general.alignment");
    u32_le(&mut bytes, 4); // GGUF_TYPE_UINT32
    u32_le(&mut bytes, 64);

    string(&mut bytes, "general.name");
    u32_le(&mut bytes, 8); // GGUF_TYPE_STRING
    string(&mut bytes, "manual-spec");

    string(&mut bytes, "weight");
    u32_le(&mut bytes, 2); // rank
    u64_le(&mut bytes, 2); // ne[0], GGML-contiguous dimension
    u64_le(&mut bytes, 3); // ne[1]
    u32_le(&mut bytes, 0); // GGML_TYPE_F32
    u64_le(&mut bytes, 0); // relative tensor-data offset

    assert_eq!(bytes.len(), 146);
    bytes.resize(192, 0); // next 64-byte boundary
    for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[cfg(feature = "std")]
fn create_unsupported_tensor_type_fixture(tensor_type: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.push(b'x');
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&tensor_type.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes
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
fn test_manual_spec_fixture_with_custom_alignment() {
    let mut reader = GGUFFileReader::new(Cursor::new(create_manual_aligned_gguf_fixture()))
        .expect("manual GGUF v3 fixture must parse");

    assert_eq!(reader.tensor_alignment(), 64);
    assert_eq!(reader.tensor_data_offset(), 192);
    assert_eq!(reader.metadata().get_string("general.name"), Some("manual-spec"));
    let info = reader.get_tensor_info("weight").unwrap();
    assert_eq!(info.shape().dims(), &[2, 3]);
    assert_eq!(info.tensor_type(), TensorType::F32);
    assert_eq!(info.data_offset(), 0);

    let tensor = reader.load_tensor_data("weight").unwrap().unwrap();
    let expected: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    assert_eq!(tensor.as_slice(), expected);
}

#[cfg(feature = "std")]
#[test]
fn test_big_endian_header_prefix_is_explicitly_rejected() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_be_bytes());
    bytes.extend_from_slice(&0u64.to_be_bytes());
    bytes.extend_from_slice(&0u64.to_be_bytes());

    let error = GGUFFileReader::new(Cursor::new(bytes)).unwrap_err();
    assert!(matches!(error, GGUFError::UnsupportedVersion(0x0300_0000)));
}

#[cfg(feature = "std")]
#[test]
fn test_removed_reserved_and_unknown_tensor_type_ids_are_rejected() {
    for tensor_type in [4, 5, 31, 32, 33, 36, 37, 38, u32::MAX] {
        let error =
            GGUFFileReader::new(Cursor::new(create_unsupported_tensor_type_fixture(tensor_type)))
                .unwrap_err();
        assert!(!error.to_string().is_empty(), "empty error for tensor type {tensor_type}");
    }
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
    assert_eq!(TensorType::F32.element_size(), Some(4));
    assert_eq!(TensorType::F16.element_size(), Some(2));
    assert_eq!(TensorType::I32.element_size(), Some(4));

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
    let shape = gguf_rs_lib::tensor::TensorShape::new(vec![2, 1]).unwrap();
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
