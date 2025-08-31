//! Test fixtures for GGUF tests

use gguf::prelude::*;
use std::io::Write;
use tempfile::NamedTempFile;

/// Create a minimal valid GGUF file with no tensors or metadata
pub fn create_minimal_gguf() -> Vec<u8> {
    let mut data = Vec::new();
    
    // GGUF header
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count
    
    data
}

/// Create a simple GGUF file with basic metadata and one tensor
pub fn create_simple_gguf() -> Vec<u8> {
    let builder = GGUFBuilder::simple("test_model", "A simple test model")
        .add_f32_tensor("weights", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    
    builder.build_to_bytes().expect("Failed to build simple GGUF")
}

/// Create a GGUF file with comprehensive metadata
pub fn create_metadata_rich_gguf() -> Vec<u8> {
    let mut builder = GGUFBuilder::simple("metadata_test", "Testing metadata");
    
    builder = builder
        .add_metadata("test.uint8", MetadataValue::UInt8(255))
        .add_metadata("test.int8", MetadataValue::Int8(-128))
        .add_metadata("test.uint16", MetadataValue::UInt16(65535))
        .add_metadata("test.int16", MetadataValue::Int16(-32768))
        .add_metadata("test.uint32", MetadataValue::UInt32(4294967295))
        .add_metadata("test.int32", MetadataValue::Int32(-2147483648))
        .add_metadata("test.uint64", MetadataValue::UInt64(18446744073709551615))
        .add_metadata("test.int64", MetadataValue::Int64(-9223372036854775808))
        .add_metadata("test.float32", MetadataValue::Float32(3.14159))
        .add_metadata("test.float64", MetadataValue::Float64(2.71828))
        .add_metadata("test.bool_true", MetadataValue::Bool(true))
        .add_metadata("test.bool_false", MetadataValue::Bool(false))
        .add_metadata("test.string", MetadataValue::String("Hello, World!".to_string()))
        .add_metadata("test.array", MetadataValue::Array(vec![
            MetadataValue::UInt32(1),
            MetadataValue::UInt32(2),
            MetadataValue::UInt32(3),
        ]));
    
    builder.build_to_bytes().expect("Failed to build metadata-rich GGUF")
}

/// Create a GGUF file with multiple tensor types
pub fn create_multi_tensor_gguf() -> Vec<u8> {
    let mut builder = GGUFBuilder::simple("multi_tensor", "Multiple tensor types");
    
    builder = builder
        .add_f32_tensor("f32_tensor", vec![3, 3], vec![1.0; 9])
        .add_i32_tensor("i32_tensor", vec![4], vec![10, 20, 30, 40])
        .add_quantized_tensor(
            "q4_tensor", 
            vec![64], // 2 blocks of 32 elements each
            TensorType::Q4_0, 
            vec![0u8; 36] // 2 * 18 bytes per block
        )
        .add_quantized_tensor(
            "q8_tensor",
            vec![32], // 1 block of 32 elements
            TensorType::Q8_0,
            vec![0u8; 34] // 1 * 34 bytes per block
        );
    
    builder.build_to_bytes().expect("Failed to build multi-tensor GGUF")
}

/// Create a GGUF file with large tensors for performance testing
pub fn create_large_gguf() -> Vec<u8> {
    let mut builder = GGUFBuilder::simple("large_model", "Large model for testing");
    
    // Create large embedding matrix (similar to real models)
    let vocab_size = 32000;
    let embed_dim = 4096;
    let embedding_data: Vec<f32> = (0..(vocab_size * embed_dim))
        .map(|i| (i as f32) * 0.0001)
        .collect();
    
    builder = builder.add_f32_tensor(
        "token_embd.weight", 
        vec![vocab_size as u64, embed_dim as u64], 
        embedding_data
    );
    
    // Add some layer weights
    for i in 0..4 {
        let layer_weights: Vec<f32> = (0..(embed_dim * embed_dim))
            .map(|j| ((i * 10000 + j) as f32) * 0.00001)
            .collect();
        
        builder = builder.add_f32_tensor(
            &format!("layers.{}.attention.weight", i),
            vec![embed_dim as u64, embed_dim as u64],
            layer_weights
        );
    }
    
    builder.build_to_bytes().expect("Failed to build large GGUF")
}

/// Create a GGUF file with edge case tensors (empty, unusual shapes)
pub fn create_edge_case_gguf() -> Vec<u8> {
    let mut builder = GGUFBuilder::simple("edge_cases", "Edge case tensors");
    
    builder = builder
        // Empty tensors
        .add_f32_tensor("empty_1d", vec![0], vec![])
        .add_f32_tensor("empty_2d", vec![0, 5], vec![])
        .add_f32_tensor("empty_3d", vec![2, 0, 3], vec![])
        
        // Single element tensors
        .add_f32_tensor("scalar", vec![1], vec![42.0])
        .add_i32_tensor("single_int", vec![1], vec![100])
        
        // High-dimensional tensors
        .add_f32_tensor("high_dim", vec![2, 2, 2, 2, 2], vec![1.0; 32])
        
        // Very wide tensor
        .add_f32_tensor("wide", vec![1000], (0..1000).map(|i| i as f32).collect())
        
        // Very tall tensor  
        .add_f32_tensor("tall", vec![500, 1], vec![0.5; 500]);
    
    builder.build_to_bytes().expect("Failed to build edge case GGUF")
}

/// Create an invalid GGUF file with wrong magic number
pub fn create_invalid_magic_gguf() -> Vec<u8> {
    let mut data = Vec::new();
    
    // Wrong magic number
    data.extend_from_slice(&0x12345678u32.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    
    data
}

/// Create an invalid GGUF file with unsupported version
pub fn create_invalid_version_gguf() -> Vec<u8> {
    let mut data = Vec::new();
    
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&999u32.to_le_bytes()); // Unsupported version
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    
    data
}

/// Create a truncated GGUF file (incomplete header)
pub fn create_truncated_gguf() -> Vec<u8> {
    vec![0x47, 0x47, 0x55] // Only 3 bytes
}

/// Create a GGUF file with corrupted metadata
pub fn create_corrupted_metadata_gguf() -> Vec<u8> {
    let mut data = Vec::new();
    
    // Valid header claiming 1 metadata entry
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count
    
    // Corrupted metadata: claim huge key length but provide small data
    data.extend_from_slice(&1000u64.to_le_bytes()); // key length
    data.extend_from_slice(b"short"); // But only provide short data
    
    data
}

/// Save GGUF data to a temporary file and return the file
pub fn save_to_temp_file(data: &[u8]) -> NamedTempFile {
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file.write_all(data).expect("Failed to write test data");
    temp_file.flush().expect("Failed to flush temp file");
    temp_file
}

/// Create a GGUF file with specific alignment challenges
pub fn create_alignment_test_gguf() -> Vec<u8> {
    let mut builder = GGUFBuilder::new();
    
    // Add metadata entries of varying sizes to create alignment challenges
    for i in 0..20 {
        let key = format!("key_{}", i);
        let value = format!("value_with_length_{}", i * 3); // Varying lengths
        builder = builder.add_metadata(&key, MetadataValue::String(value));
    }
    
    // Add tensors of different sizes
    builder = builder
        .add_f32_tensor("tensor_1", vec![1], vec![1.0]) // 4 bytes
        .add_f32_tensor("tensor_3", vec![3], vec![1.0, 2.0, 3.0]) // 12 bytes
        .add_f32_tensor("tensor_7", vec![7], vec![1.0; 7]) // 28 bytes
        .add_f32_tensor("tensor_13", vec![13], vec![1.0; 13]) // 52 bytes
        .add_i32_tensor("int_tensor_5", vec![5], vec![1, 2, 3, 4, 5]) // 20 bytes
        .add_i32_tensor("int_tensor_11", vec![11], (0..11).collect()); // 44 bytes
    
    builder.build_to_bytes().expect("Failed to build alignment test GGUF")
}

/// Test data for quantization tests
pub fn create_quantization_test_data() -> Vec<f32> {
    // Create data with known patterns for quantization testing
    let mut data = Vec::new();
    
    // Add various value ranges
    data.extend((0..32).map(|i| i as f32)); // 0-31
    data.extend((0..32).map(|i| (i as f32) * 0.1)); // 0-3.1 by 0.1
    data.extend((0..32).map(|i| (i as f32) * -0.1)); // 0 to -3.1 by -0.1
    data.extend((0..32).map(|i| (i as f32) * 100.0)); // 0-3100 by 100
    
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use gguf::reader::GGUFFileReader;
    use std::io::Cursor;
    
    #[test]
    fn test_minimal_fixture() {
        let data = create_minimal_gguf();
        assert!(!data.is_empty());
        
        let cursor = Cursor::new(data);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read minimal GGUF");
        
        assert_eq!(reader.tensor_count(), 0);
        assert_eq!(reader.metadata().len(), 0);
    }
    
    #[test]
    fn test_simple_fixture() {
        let data = create_simple_gguf();
        assert!(!data.is_empty());
        
        let cursor = Cursor::new(data);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read simple GGUF");
        
        assert_eq!(reader.tensor_count(), 1);
        assert_eq!(reader.metadata().len(), 2); // name + description
        assert_eq!(reader.metadata().get_string("general.name"), Some("test_model"));
    }
    
    #[test]
    fn test_metadata_rich_fixture() {
        let data = create_metadata_rich_gguf();
        assert!(!data.is_empty());
        
        let cursor = Cursor::new(data);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read metadata-rich GGUF");
        
        assert!(reader.metadata().len() > 10);
        assert_eq!(reader.metadata().get_string("test.string"), Some("Hello, World!"));
        assert_eq!(reader.metadata().get_bool("test.bool_true"), Some(true));
        assert_eq!(reader.metadata().get_f32("test.float32"), Some(3.14159));
    }
    
    #[test]
    fn test_multi_tensor_fixture() {
        let data = create_multi_tensor_gguf();
        assert!(!data.is_empty());
        
        let cursor = Cursor::new(data);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read multi-tensor GGUF");
        
        assert_eq!(reader.tensor_count(), 4);
        
        let f32_info = reader.get_tensor_info("f32_tensor").unwrap();
        assert_eq!(f32_info.tensor_type(), TensorType::F32);
        
        let q4_info = reader.get_tensor_info("q4_tensor").unwrap();
        assert_eq!(q4_info.tensor_type(), TensorType::Q4_0);
    }
    
    #[test]
    fn test_invalid_fixtures() {
        // Test invalid magic
        let invalid_magic = create_invalid_magic_gguf();
        let cursor = Cursor::new(invalid_magic);
        let result = GGUFFileReader::new(cursor);
        assert!(matches!(result, Err(GGUFError::InvalidMagic { .. })));
        
        // Test invalid version
        let invalid_version = create_invalid_version_gguf();
        let cursor = Cursor::new(invalid_version);
        let result = GGUFFileReader::new(cursor);
        assert!(matches!(result, Err(GGUFError::UnsupportedVersion(999))));
        
        // Test truncated
        let truncated = create_truncated_gguf();
        let cursor = Cursor::new(truncated);
        let result = GGUFFileReader::new(cursor);
        assert!(result.is_err());
    }
}