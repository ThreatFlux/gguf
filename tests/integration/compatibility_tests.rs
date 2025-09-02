//! Compatibility and edge case tests

#[cfg(feature = "std")]
use gguf_rs::builder::GGUFBuilder;
#[cfg(feature = "std")]
use gguf_rs::format::metadata::MetadataValue;
#[cfg(feature = "std")]
use gguf_rs::reader::GGUFFileReader;
#[cfg(feature = "std")]
use gguf_rs::tensor::TensorType;
#[cfg(feature = "std")]
use std::io::Cursor;

#[cfg(feature = "std")]
#[test]
fn test_empty_file_compatibility() {
    // Test creating and reading an empty GGUF file
    let builder = GGUFBuilder::new();
    let (bytes, write_result) = builder.build_to_bytes().expect("Failed to build empty file");

    assert!(!bytes.is_empty());
    // The GGUFWriteResult doesn't have tensor_count/metadata_count fields
    // These are tracked separately in the builder or need to be calculated
    assert_eq!(write_result.tensor_results.len(), 0);

    // Read it back
    let cursor = Cursor::new(bytes);
    let reader = GGUFFileReader::new(cursor).expect("Failed to read empty file");

    assert_eq!(reader.tensor_count(), 0);
    assert_eq!(reader.metadata().len(), 0);
}

#[cfg(feature = "std")]
#[test]
fn test_minimal_valid_file() {
    // Test the smallest possible valid GGUF file with some content
    let mut builder = GGUFBuilder::new();
    builder = builder.add_metadata("name", MetadataValue::String("minimal".to_string()));

    let data = vec![0u8; 4]; // One F32 value
    let result = builder.add_tensor("single", vec![1], TensorType::F32, data);
    assert!(result.is_ok());

    let builder = result.unwrap();
    let (bytes, write_result) = builder.build_to_bytes().expect("Failed to build minimal file");

    assert!(!bytes.is_empty());
    assert_eq!(write_result.tensor_results.len(), 1);

    // Verify it can be read
    let cursor = Cursor::new(bytes);
    let reader = GGUFFileReader::new(cursor).expect("Failed to read minimal file");

    assert_eq!(reader.tensor_count(), 1);
    assert_eq!(reader.metadata().len(), 1);
}

#[cfg(feature = "std")]
#[test]
fn test_special_characters_in_names() {
    // Test tensor and metadata names with special characters
    let mut builder = GGUFBuilder::new();

    // Add metadata with various name patterns
    builder = builder
        .add_metadata("simple_name", MetadataValue::String("value".to_string()))
        .add_metadata("name.with.dots", MetadataValue::U32(1))
        .add_metadata("name-with-dashes", MetadataValue::U32(2))
        .add_metadata("name_with_underscores", MetadataValue::U32(3));

    // Add tensors with various name patterns
    let data = vec![0u8; 4];
    let result = builder
        .add_tensor("tensor.with.dots", vec![1], TensorType::F32, data.clone())
        .unwrap()
        .add_tensor("tensor-with-dashes", vec![1], TensorType::F32, data.clone())
        .unwrap()
        .add_tensor("tensor_with_underscores", vec![1], TensorType::F32, data)
        .unwrap();

    let (bytes, _) = result.build_to_bytes().expect("Failed to build with special names");

    // Verify it can be read
    let cursor = Cursor::new(bytes);
    let reader =
        gguf_rs::reader::GGUFFileReader::new(cursor).expect("Failed to read special names");

    assert!(reader.metadata().contains_key("name.with.dots"));
    assert!(reader.get_tensor_info("tensor.with.dots").is_some());
}

#[cfg(feature = "std")]
#[test]
fn test_zero_dimensional_edge_cases() {
    // Test tensors with zero in some dimensions
    let builder = GGUFBuilder::new();

    // Empty tensor (0 elements)
    let empty_data = vec![];
    let result = builder.add_tensor("empty", vec![0], TensorType::F32, empty_data);
    assert!(result.is_ok());

    let builder = result.unwrap();
    let (bytes, write_result) =
        builder.build_to_bytes().expect("Failed to build with empty tensor");

    assert!(!bytes.is_empty());
    assert_eq!(write_result.tensor_results.len(), 1);

    // Read back and verify
    let cursor = Cursor::new(bytes);
    let reader = GGUFFileReader::new(cursor).expect("Failed to read empty tensor");

    let tensor_info = reader.get_tensor_info("empty").unwrap();
    assert_eq!(tensor_info.element_count(), 0);
}

#[cfg(feature = "std")]
#[test]
fn test_unicode_string_handling() {
    // Test Unicode strings in metadata
    let mut builder = GGUFBuilder::new();

    builder = builder
        .add_metadata("english", MetadataValue::String("Hello World".to_string()))
        .add_metadata("chinese", MetadataValue::String("你好世界".to_string()))
        .add_metadata("emoji", MetadataValue::String("🦀🚀💯".to_string()))
        .add_metadata(
            "mixed",
            MetadataValue::String("Mixed: ASCII + 中文 + العربية + 🌍".to_string()),
        );

    let (bytes, _) = builder.build_to_bytes().expect("Failed to build with Unicode");

    // Read back and verify
    let cursor = Cursor::new(bytes);
    let reader = gguf_rs::reader::GGUFFileReader::new(cursor).expect("Failed to read Unicode");

    assert_eq!(reader.metadata().get_string("english"), Some("Hello World"));
    assert_eq!(reader.metadata().get_string("chinese"), Some("你好世界"));
    assert_eq!(reader.metadata().get_string("emoji"), Some("🦀🚀💯"));
    assert!(reader.metadata().get_string("mixed").unwrap().contains("🌍"));
}
