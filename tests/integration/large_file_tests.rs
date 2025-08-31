//! Large file handling tests

use gguf::builder::GGUFBuilder;
use gguf::format::metadata::MetadataValue;
use gguf::tensor::TensorType;

#[test]
fn test_large_tensor_handling() {
    // Test handling of reasonably large tensors
    let size = 10_000; // 10K elements to keep test reasonable
    let data = vec![0u8; size * 4]; // F32 data

    let builder = GGUFBuilder::new();
    let result = builder.add_tensor("large_tensor", vec![size as u64], TensorType::F32, data);

    assert!(result.is_ok());
    let builder = result.unwrap();

    let (bytes, write_result) = builder.build_to_bytes().expect("Failed to build large tensor");

    assert!(!bytes.is_empty());
    assert!(write_result.total_bytes_written > size * 4);
}

#[test]
fn test_many_small_tensors() {
    // Test handling many small tensors
    let mut builder = GGUFBuilder::new();

    for i in 0..100 {
        let tensor_name = format!("tensor_{}", i);
        let data = vec![i as u8; 4]; // Small tensor data

        let result = builder.add_tensor(&tensor_name, vec![1], TensorType::F32, data);
        assert!(result.is_ok());
        builder = result.unwrap();
    }

    assert_eq!(builder.tensor_count(), 100);

    let (bytes, write_result) = builder.build_to_bytes().expect("Failed to build many tensors");

    assert!(!bytes.is_empty());
    assert_eq!(write_result.tensor_results.len(), 100);
}

#[test]
fn test_large_metadata() {
    // Test handling of large amounts of metadata
    let mut builder = GGUFBuilder::new();

    for i in 0..1000 {
        let key = format!("metadata_key_{}", i);
        let value = format!("This is metadata value number {}", i);
        builder = builder.add_metadata(key, MetadataValue::String(value));
    }

    assert_eq!(builder.metadata_count(), 1000);

    let (bytes, write_result) = builder.build_to_bytes().expect("Failed to build large metadata");

    assert!(!bytes.is_empty());
    assert!(write_result.metadata_result.bytes_written > 10000); // Should be substantial
}
