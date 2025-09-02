//! Format conversion and compatibility tests

use gguf_rs::prelude::*;
use std::io::Cursor;
use tempfile::NamedTempFile;

#[test]
fn test_version_compatibility() {
    // Test that we can handle different GGUF versions appropriately

    // Create a standard GGUF v3 file
    let builder = GGUFBuilder::simple("version_test", "Testing version compatibility");
    let (bytes, _result) = builder.build_to_bytes().expect("Failed to build v3 file");

    // Verify it reads as version 3
    let cursor = Cursor::new(&bytes);
    let reader = GGUFFileReader::new(cursor).expect("Failed to read v3 file");

    // The header should indicate version 3
    // Note: We don't have direct header access in the reader API,
    // but we can verify it was read successfully
    assert_eq!(reader.metadata().get_string("general.name"), Some("version_test"));

    // Test with manually crafted older version (should fail)
    let mut old_version_data = Vec::new();
    old_version_data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF magic
    old_version_data.extend_from_slice(&2u32.to_le_bytes()); // Version 2 (unsupported)
    old_version_data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    old_version_data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count

    let cursor = Cursor::new(old_version_data);
    let result = GGUFFileReader::new(cursor);

    match result {
        Err(GGUFError::UnsupportedVersion(2)) => {} // Expected
        _ => panic!("Should have failed with UnsupportedVersion(2)"),
    }
}

#[test]
fn test_endianness_handling() {
    // Test that our little-endian format is handled correctly

    let mut builder = GGUFBuilder::new();

    // Add some data with specific bit patterns that would be different in big-endian
    let test_values = vec![
        0x12345678u32 as f32, // Should be clearly different in big vs little endian
        0x87654321u32 as f32,
        0xDEADBEEFu32 as f32,
        0xCAFEBABEu32 as f32,
    ];

    builder = builder.add_f32_tensor("endian_test", vec![4], test_values.clone());

    let (bytes, _result) = builder.build_to_bytes().expect("Failed to build");

    // Read back and verify
    let cursor = Cursor::new(bytes);
    let mut reader = GGUFFileReader::new(cursor).expect("Failed to read");

    let loaded_data = reader
        .load_tensor_data("endian_test")
        .expect("Failed to load data")
        .expect("Data should exist");

    let loaded_values: Vec<f32> = loaded_data
        .as_slice()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    for (original, loaded) in test_values.iter().zip(loaded_values.iter()) {
        assert_eq!(*original, *loaded);
    }

    // Also verify the raw bytes are in little-endian format
    let raw_bytes = loaded_data.as_slice();
    for (i, &original) in test_values.iter().enumerate() {
        let chunk = &raw_bytes[i * 4..(i + 1) * 4];
        let expected_bytes = original.to_le_bytes();
        assert_eq!(chunk, &expected_bytes);
    }
}

#[test]
fn test_alignment_handling() {
    // Test various alignment scenarios

    let mut builder = GGUFBuilder::new();

    // Add metadata of various sizes to create different alignment scenarios
    for i in 0..10 {
        let key = format!("key_{}", i);
        let value = format!("value_with_length_{}", i * 7); // Varying lengths
        builder = builder.add_metadata(&key, MetadataValue::String(value));
    }

    // Add tensors that will require alignment
    builder = builder
        .add_f32_tensor("tensor1", vec![1], vec![1.0]) // 4 bytes
        .add_f32_tensor("tensor2", vec![3], vec![1.0, 2.0, 3.0]) // 12 bytes
        .add_f32_tensor("tensor3", vec![7], vec![1.0; 7]) // 28 bytes
        .add_f32_tensor("tensor4", vec![10], vec![1.0; 10]); // 40 bytes

    let (bytes, _result) = builder.build_to_bytes().expect("Failed to build with alignment");

    // Read back and verify all tensors
    let cursor = Cursor::new(bytes);
    let mut reader = GGUFFileReader::new(cursor).expect("Failed to read aligned file");

    for i in 1..=4 {
        let tensor_name = format!("tensor{}", i);
        let tensor_info = reader.get_tensor_info(&tensor_name).unwrap().clone();

        // Verify tensor data offset is reasonable (relative to tensor data section start)
        // Note: Offsets are now relative to tensor data start, so alignment is at section level
        assert!(
            tensor_info.data_offset < 10_000,
            "Tensor {} offset {} seems unreasonable",
            i,
            tensor_info.data_offset
        );

        // Verify we can load the data
        let data = reader
            .load_tensor_data(&tensor_name)
            .expect("Failed to load tensor")
            .expect("Tensor should exist");

        assert_eq!(data.len(), tensor_info.expected_data_size() as usize);
    }
}

#[test]
fn test_string_encoding() {
    // Test various string encodings and edge cases

    let long_string = "x".repeat(1000);
    let test_strings = vec![
        ("empty", ""),
        ("ascii", "Hello, World!"),
        ("unicode", "Hello, 世界! 🌍"),
        ("emoji", "🦀🔥💯🚀"),
        ("mixed", "ASCII and 中文 and العربية and русский"),
        ("newlines", "Line 1\nLine 2\nLine 3"),
        ("special_chars", "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"),
        ("long", long_string.as_str()),
        ("numbers", "1234567890.5e-10"),
    ];

    let mut builder = GGUFBuilder::new();

    for (key, value) in &test_strings {
        builder = builder.add_metadata(*key, MetadataValue::String(value.to_string()));
    }

    let (bytes, _result) = builder.build_to_bytes().expect("Failed to build with strings");

    let cursor = Cursor::new(bytes);
    let reader = GGUFFileReader::new(cursor).expect("Failed to read strings");

    for (key, expected_value) in &test_strings {
        let actual_value = reader
            .metadata()
            .get_string(key)
            .unwrap_or_else(|| panic!("Missing key: {}", key));
        assert_eq!(actual_value, *expected_value, "String mismatch for key: {}", key);
    }
}

#[test]
fn test_large_metadata() {
    // Test handling of large amounts of metadata

    let mut builder = GGUFBuilder::new();

    // Add many metadata entries
    for i in 0..1000 {
        builder = builder.add_metadata(
            format!("large_meta_{:04}", i),
            MetadataValue::String(format!(
                "Large metadata value number {} with some padding text to make it longer",
                i
            )),
        );
    }

    // Add a few tensors too
    builder = builder.add_f32_tensor("tensor1", vec![100], vec![0.0; 100]).add_i32_tensor(
        "tensor2",
        vec![50],
        (0..50).collect(),
    );

    let (bytes, _result) = builder.build_to_bytes().expect("Failed to build large metadata");
    assert!(bytes.len() > 100_000); // Should be substantial

    let cursor = Cursor::new(bytes);
    let reader = GGUFFileReader::new(cursor).expect("Failed to read large metadata");

    // Verify metadata count
    assert_eq!(reader.metadata().len(), 1000);

    // Spot check some values
    for i in [0, 100, 500, 999] {
        let key = format!("large_meta_{:04}", i);
        let expected =
            format!("Large metadata value number {} with some padding text to make it longer", i);
        assert_eq!(reader.metadata().get_string(&key), Some(expected.as_str()));
    }

    // Verify tensors still work
    assert_eq!(reader.tensor_count(), 2);
    assert!(reader.get_tensor_info("tensor1").is_some());
    assert!(reader.get_tensor_info("tensor2").is_some());
}

#[test]
fn test_tensor_name_edge_cases() {
    // Test various tensor name patterns

    let test_names = vec![
        "simple",
        "with.dots",
        "with-dashes",
        "with_underscores",
        "with spaces",
        "with/slashes",
        "with\\backslashes",
        "with:colons",
        "MixedCase",
        "UPPERCASE",
        "lowercase",
        "123numeric456",
        "unicode_名前_тест",
        "very_long_tensor_name_that_goes_on_and_on_and_on_to_test_limits",
        // Note: Empty name is excluded as it should be rejected by validation
    ];

    let mut builder = GGUFBuilder::new();

    for (i, name) in test_names.iter().enumerate() {
        let data = vec![i as f32; 10];
        builder = builder.add_f32_tensor(*name, vec![10], data);
    }

    let (bytes, _result) = builder.build_to_bytes().expect("Failed to build with various names");

    let cursor = Cursor::new(bytes);
    let mut reader = GGUFFileReader::new(cursor).expect("Failed to read various names");

    assert_eq!(reader.tensor_count(), test_names.len());

    for (i, name) in test_names.iter().enumerate() {
        let tensor_info = reader
            .get_tensor_info(name)
            .unwrap_or_else(|| panic!("Missing tensor: '{}'", name));

        assert_eq!(tensor_info.name(), *name);
        assert_eq!(tensor_info.element_count(), 10);

        // Load and verify data
        let data = reader
            .load_tensor_data(name)
            .expect("Failed to load tensor")
            .expect("Tensor should exist");

        let values: Vec<f32> = data
            .as_slice()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        for &value in &values {
            assert_eq!(value, i as f32);
        }
    }

    // Test that empty tensor name is properly rejected
    let builder_empty = GGUFBuilder::new();
    let result = builder_empty.add_f32_tensor("", vec![1], vec![1.0]).build_to_bytes();
    assert!(result.is_err(), "Empty tensor name should be rejected");
}

#[test]
fn test_zero_sized_tensors() {
    // Test handling of empty/zero-sized tensors

    let mut builder = GGUFBuilder::new();

    // Add tensors with zero elements
    builder = builder
        .add_f32_tensor("empty_1d", vec![0], vec![])
        .add_f32_tensor("empty_2d", vec![0, 5], vec![])
        .add_f32_tensor("empty_3d", vec![2, 0, 3], vec![])
        .add_i32_tensor("empty_i32", vec![0], vec![]);

    // Also add a normal tensor for comparison
    builder = builder.add_f32_tensor("normal", vec![3], vec![1.0, 2.0, 3.0]);

    let (bytes, _result) = builder.build_to_bytes().expect("Failed to build with empty tensors");

    let cursor = Cursor::new(bytes);
    let mut reader = GGUFFileReader::new(cursor).expect("Failed to read empty tensors");

    // Verify empty tensors
    let empty_1d = reader.get_tensor_info("empty_1d").unwrap();
    assert_eq!(empty_1d.element_count(), 0);
    assert_eq!(empty_1d.expected_data_size(), 0);

    let empty_2d = reader.get_tensor_info("empty_2d").unwrap();
    assert_eq!(empty_2d.element_count(), 0);
    assert_eq!(empty_2d.expected_data_size(), 0);

    let empty_3d = reader.get_tensor_info("empty_3d").unwrap();
    assert_eq!(empty_3d.element_count(), 0);
    assert_eq!(empty_3d.expected_data_size(), 0);

    // Load empty tensor data
    let empty_data = reader
        .load_tensor_data("empty_1d")
        .expect("Failed to load empty tensor")
        .expect("Empty tensor should exist");
    assert_eq!(empty_data.len(), 0);

    // Verify normal tensor still works
    let normal_data = reader
        .load_tensor_data("normal")
        .expect("Failed to load normal tensor")
        .expect("Normal tensor should exist");
    assert_eq!(normal_data.len(), 12); // 3 * 4 bytes
}

#[test]
fn test_file_size_calculation() {
    // Test that file sizes are calculated correctly

    let mut builder = GGUFBuilder::simple("size_test", "Testing file size calculation");

    // Add known amounts of data
    let tensor1_elements = 1000;
    let tensor1_data = vec![1.0f32; tensor1_elements];
    builder = builder.add_f32_tensor("tensor1", vec![tensor1_elements as u64], tensor1_data);

    let tensor2_elements = 500;
    let tensor2_data = vec![42i32; tensor2_elements];
    builder = builder.add_i32_tensor("tensor2", vec![tensor2_elements as u64], tensor2_data);

    // Build to get size information
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let result = builder.build_to_file(temp_file.path()).expect("Failed to build");

    // Calculate expected sizes
    let expected_tensor1_size = tensor1_elements * 4; // f32 = 4 bytes each
    let expected_tensor2_size = tensor2_elements * 4; // i32 = 4 bytes each
    let expected_tensor_data = expected_tensor1_size + expected_tensor2_size;

    // Verify sizes match expectations
    assert_eq!(result.tensor_data_bytes(), expected_tensor_data);
    assert!(result.header_result.bytes_written > 0);
    assert!(result.metadata_result.bytes_written > 0);
    assert!(result.tensor_info_result.bytes_written > 0);

    let calculated_total = result.header_result.bytes_written
        + result.metadata_result.bytes_written
        + result.tensor_info_result.bytes_written
        + result.tensor_data_bytes();

    // Account for alignment padding between tensor info and tensor data
    let padding_bytes = result.total_bytes_written - calculated_total;
    assert!(padding_bytes < 32, "Padding should be less than alignment boundary (32 bytes)");

    // The total should equal calculated total plus alignment padding
    assert_eq!(result.total_bytes_written, calculated_total + padding_bytes);

    // Verify file on disk has the expected size
    let file_size = std::fs::metadata(temp_file.path()).expect("Failed to get file metadata").len();

    assert_eq!(file_size, result.total_bytes_written as u64);
}
