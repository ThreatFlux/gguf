//! Unit tests for error handling

use gguf::prelude::*;
use std::io::{self, Cursor};
use std::fmt;

mod gguf_error_tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let errors = vec![
            GGUFError::InvalidMagic { expected: 0x46554747, found: 0x12345678 },
            GGUFError::UnsupportedVersion(999),
            GGUFError::InvalidTensorType(42),
            GGUFError::InvalidMetadataType(13),
            GGUFError::TensorNotFound("missing_tensor".to_string()),
            GGUFError::InvalidTensorData("corrupted data".to_string()),
            GGUFError::AlignmentError { expected: 32, actual: 17 },
            GGUFError::IO(io::Error::new(io::ErrorKind::UnexpectedEof, "test error")),
        ];

        for error in errors {
            let display_str = format!("{}", error);
            assert!(!display_str.is_empty());
            
            let debug_str = format!("{:?}", error);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_error_source() {
        let io_error = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
        let gguf_error = GGUFError::IO(io_error);
        
        assert!(gguf_error.source().is_some());
        
        let other_error = GGUFError::UnsupportedVersion(2);
        assert!(other_error.source().is_none());
    }

    #[test]
    fn test_error_from_io_error() {
        let io_error = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let gguf_error: GGUFError = io_error.into();
        
        match gguf_error {
            GGUFError::IO(inner) => {
                assert_eq!(inner.kind(), io::ErrorKind::NotFound);
            }
            _ => panic!("Expected IO error variant"),
        }
    }

    #[test]
    fn test_invalid_magic_error() {
        let error = GGUFError::InvalidMagic { expected: 0x46554747, found: 0x47474546 };
        
        let message = format!("{}", error);
        assert!(message.contains("0x46554747"));
        assert!(message.contains("0x47474546"));
    }

    #[test]
    fn test_unsupported_version_error() {
        let error = GGUFError::UnsupportedVersion(5);
        
        let message = format!("{}", error);
        assert!(message.contains("5"));
        assert!(message.contains("version"));
    }

    #[test]
    fn test_invalid_tensor_type_error() {
        let error = GGUFError::InvalidTensorType(999);
        
        let message = format!("{}", error);
        assert!(message.contains("999"));
        assert!(message.contains("tensor type"));
    }

    #[test]
    fn test_invalid_metadata_type_error() {
        let error = GGUFError::InvalidMetadataType(50);
        
        let message = format!("{}", error);
        assert!(message.contains("50"));
        assert!(message.contains("metadata type"));
    }

    #[test]
    fn test_tensor_not_found_error() {
        let tensor_name = "nonexistent_tensor";
        let error = GGUFError::TensorNotFound(tensor_name.to_string());
        
        let message = format!("{}", error);
        assert!(message.contains(tensor_name));
    }

    #[test]
    fn test_invalid_tensor_data_error() {
        let description = "data size mismatch";
        let error = GGUFError::InvalidTensorData(description.to_string());
        
        let message = format!("{}", error);
        assert!(message.contains(description));
    }

    #[test]
    fn test_alignment_error() {
        let error = GGUFError::AlignmentError { expected: 32, actual: 19 };
        
        let message = format!("{}", error);
        assert!(message.contains("32"));
        assert!(message.contains("19"));
        assert!(message.contains("alignment"));
    }

    #[test]
    fn test_error_categories() {
        // Test that errors are properly categorized
        assert!(matches!(
            GGUFError::InvalidMagic { expected: 0, found: 1 },
            GGUFError::InvalidMagic { .. }
        ));
        
        assert!(matches!(
            GGUFError::UnsupportedVersion(1),
            GGUFError::UnsupportedVersion(_)
        ));
        
        assert!(matches!(
            GGUFError::InvalidTensorType(1),
            GGUFError::InvalidTensorType(_)
        ));
    }

    #[test]
    fn test_result_type_alias() {
        // Test that our Result type alias works correctly
        fn test_function() -> Result<i32> {
            Ok(42)
        }
        
        fn test_function_err() -> Result<i32> {
            Err(GGUFError::UnsupportedVersion(99))
        }
        
        assert_eq!(test_function().unwrap(), 42);
        assert!(test_function_err().is_err());
        
        match test_function_err() {
            Err(GGUFError::UnsupportedVersion(99)) => {}
            _ => panic!("Expected UnsupportedVersion error"),
        }
    }

    #[test]
    fn test_error_chain() {
        // Test error chaining through multiple layers
        let io_error = io::Error::new(io::ErrorKind::UnexpectedEof, "EOF while reading header");
        let gguf_error = GGUFError::IO(io_error);
        
        // Should be able to downcast to original IO error
        match &gguf_error {
            GGUFError::IO(inner) => {
                assert_eq!(inner.kind(), io::ErrorKind::UnexpectedEof);
                assert!(inner.to_string().contains("EOF"));
            }
            _ => panic!("Expected IO error"),
        }
    }
}

mod error_propagation_tests {
    use super::*;
    use std::io::Write;

    // Test error propagation through the library stack
    
    #[test]
    fn test_reader_error_propagation() {
        // Create invalid GGUF data
        let invalid_data = vec![0x12, 0x34, 0x56, 0x78]; // Wrong magic number
        let cursor = Cursor::new(invalid_data);
        
        let result = gguf::reader::GGUFFileReader::new(cursor);
        
        match result {
            Err(GGUFError::InvalidMagic { expected, found }) => {
                assert_eq!(expected, 0x46554747); // "GGUF"
                assert_eq!(found, 0x78563412); // Little-endian interpretation
            }
            _ => panic!("Expected InvalidMagic error"),
        }
    }

    #[test]
    fn test_writer_error_propagation() {
        // Create a cursor that will fail on write
        struct FailingWriter;
        
        impl io::Write for FailingWriter {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::NoSpaceLeft, "disk full"))
            }
            
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        
        impl io::Seek for FailingWriter {
            fn seek(&mut self, _pos: io::SeekFrom) -> io::Result<u64> {
                Ok(0)
            }
        }
        
        let writer = FailingWriter;
        let result = gguf::writer::GGUFFileWriter::new(writer);
        
        // The writer creation itself shouldn't fail, but writing should
        let mut file_writer = result.expect("Writer creation should succeed");
        
        let header = GGUFHeader::default();
        let write_result = file_writer.write_header(&header);
        
        match write_result {
            Err(GGUFError::IO(inner)) => {
                assert_eq!(inner.kind(), io::ErrorKind::NoSpaceLeft);
            }
            _ => panic!("Expected IO error from failed write"),
        }
    }

    #[test]
    fn test_tensor_validation_errors() {
        // Test various tensor validation errors
        let test_cases = vec![
            // (data_size, expected_size, tensor_type, should_error)
            (16, 16, TensorType::F32, false), // Valid: 4 elements * 4 bytes
            (12, 16, TensorType::F32, true),  // Invalid: 3 elements but expecting 4
            (20, 16, TensorType::F32, true),  // Invalid: 5 elements but expecting 4
        ];
        
        for (data_size, expected_elements, tensor_type, should_error) in test_cases {
            let data = vec![0u8; data_size];
            let tensor_data = TensorData::Bytes(data);
            
            // Create tensor info that expects a different size
            let shape = TensorShape::new(vec![expected_elements / tensor_type.size_in_bytes() as u64]);
            let info = TensorInfo::new(
                "test".to_string(),
                tensor_type,
                shape,
                0,
            );
            
            // This validation would typically happen in the builder or writer
            let expected_bytes = info.size_in_bytes();
            let actual_bytes = tensor_data.len();
            
            if should_error {
                assert_ne!(expected_bytes, actual_bytes as u64);
            } else {
                assert_eq!(expected_bytes, actual_bytes as u64);
            }
        }
    }

    #[test]
    fn test_metadata_serialization_errors() {
        // Test metadata serialization with edge cases
        use std::collections::HashMap;
        
        let mut metadata = Metadata::new();
        
        // Add extremely long key (should work but test limits)
        let long_key = "x".repeat(65536); // 64KB key
        metadata.insert(long_key.clone(), MetadataValue::UInt32(1));
        
        // Serialization should handle this gracefully
        let result = metadata.to_bytes();
        assert!(result.is_ok());
        
        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
        
        // Should be able to deserialize
        let mut cursor = Cursor::new(&bytes);
        let deserialized_result = Metadata::read(&mut cursor, 1);
        assert!(deserialized_result.is_ok());
        
        let deserialized = deserialized_result.unwrap();
        assert_eq!(deserialized.get_u32(&long_key), Some(1));
    }

    #[test]
    fn test_truncated_data_errors() {
        // Test various truncated data scenarios
        
        // Truncated header
        let truncated_header = vec![0x47, 0x47, 0x55]; // Only 3 bytes
        let cursor = Cursor::new(truncated_header);
        let result = gguf::reader::GGUFFileReader::new(cursor);
        assert!(result.is_err());
        
        // Truncated after valid header
        let mut partial_data = Vec::new();
        partial_data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF magic
        partial_data.extend_from_slice(&3u32.to_le_bytes()); // version
        partial_data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        partial_data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count
        // Missing metadata and tensor data
        
        let cursor = Cursor::new(partial_data);
        let result = gguf::reader::GGUFFileReader::new(cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_alignment_validation_errors() {
        // Test alignment validation
        use gguf::format::{align_to, is_aligned, pad_to_alignment};
        
        // These should work
        assert_eq!(align_to(10, 8), 16);
        assert!(is_aligned(16, 8));
        assert_eq!(pad_to_alignment(10, 8), 6);
        
        // Test zero alignment (should panic or error)
        std::panic::catch_unwind(|| align_to(10, 0))
            .expect_err("Should panic on zero alignment");
    }

    #[test]
    fn test_concurrent_access_errors() {
        // Test thread safety and concurrent access patterns
        use std::sync::Arc;
        use std::thread;
        
        let data = create_test_gguf_data();
        let data = Arc::new(data);
        
        let handles: Vec<_> = (0..4).map(|_| {
            let data_clone = Arc::clone(&data);
            thread::spawn(move || {
                let cursor = Cursor::new(&**data_clone);
                gguf::reader::GGUFFileReader::new(cursor)
            })
        }).collect();
        
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.is_ok()); // All threads should succeed
        }
    }
}

// Helper function to create test GGUF data
fn create_test_gguf_data() -> Vec<u8> {
    let mut data = Vec::new();
    
    // Header
    data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF magic
    data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count
    
    data
}

mod custom_error_tests {
    use super::*;

    // Test custom error scenarios specific to GGUF
    
    #[test]
    fn test_quantization_errors() {
        // Test errors related to quantization
        
        // Invalid block size for quantized data
        let invalid_q4_data = vec![0u8; 10]; // Q4_0 needs data in multiples of 18 bytes
        let shape = TensorShape::new(vec![32]); // 32 elements = 1 block
        
        // This should be caught in validation
        let expected_size = TensorType::Q4_0.size_in_bytes() * 1; // 1 block = 18 bytes
        assert_ne!(invalid_q4_data.len(), expected_size);
        
        // Test valid quantized data
        let valid_q4_data = vec![0u8; 18]; // Exactly 1 block
        assert_eq!(valid_q4_data.len(), expected_size);
    }

    #[test]
    fn test_tensor_name_validation() {
        // Test tensor name validation scenarios
        
        // Empty name
        let empty_name = "";
        // The library should handle empty names gracefully
        
        // Very long name
        let long_name = "x".repeat(1000);
        // Should work but might have practical limits
        
        // Names with special characters
        let special_names = vec![
            "tensor/with/slashes",
            "tensor.with.dots", 
            "tensor-with-dashes",
            "tensor_with_underscores",
            "tensor with spaces",
            "tensor\nwith\nnewlines",
            "tensor\0with\0nulls",
        ];
        
        for name in special_names {
            // All should be valid as far as the format is concerned
            let info = TensorInfo::new(
                name.to_string(),
                TensorType::F32,
                TensorShape::new(vec![1]),
                0,
            );
            assert_eq!(info.name(), name);
        }
    }

    #[test]
    fn test_metadata_value_limits() {
        // Test edge cases for metadata values
        
        // Maximum values for integer types
        let max_values = vec![
            MetadataValue::UInt8(u8::MAX),
            MetadataValue::UInt16(u16::MAX),
            MetadataValue::UInt32(u32::MAX),
            MetadataValue::UInt64(u64::MAX),
            MetadataValue::Int8(i8::MIN),
            MetadataValue::Int8(i8::MAX),
            MetadataValue::Int16(i16::MIN),
            MetadataValue::Int16(i16::MAX),
            MetadataValue::Int32(i32::MIN),
            MetadataValue::Int32(i32::MAX),
            MetadataValue::Int64(i64::MIN),
            MetadataValue::Int64(i64::MAX),
        ];
        
        for value in max_values {
            let mut metadata = Metadata::new();
            metadata.insert("test".to_string(), value.clone());
            
            // Should be able to serialize and deserialize
            let bytes = metadata.to_bytes().expect("Failed to serialize");
            let mut cursor = Cursor::new(&bytes);
            let deserialized = Metadata::read(&mut cursor, 1).expect("Failed to deserialize");
            
            // Values should match
            let restored = deserialized.get("test").unwrap();
            assert_eq!(*restored, value);
        }
    }

    #[test] 
    fn test_float_special_values() {
        // Test special float values
        let special_floats = vec![
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.0f32,
            -0.0f32,
            f32::MIN,
            f32::MAX,
            f32::EPSILON,
        ];
        
        for &value in &special_floats {
            let metadata_value = MetadataValue::Float32(value);
            
            let mut metadata = Metadata::new();
            metadata.insert("float_test".to_string(), metadata_value);
            
            // Should serialize successfully
            let bytes = metadata.to_bytes().expect("Failed to serialize float");
            let mut cursor = Cursor::new(&bytes);
            let deserialized = Metadata::read(&mut cursor, 1).expect("Failed to deserialize float");
            
            let restored_value = deserialized.get_f32("float_test").unwrap();
            
            // Handle NaN specially
            if value.is_nan() {
                assert!(restored_value.is_nan());
            } else {
                assert_eq!(restored_value, value);
            }
        }
    }

    #[test]
    fn test_memory_exhaustion_protection() {
        // Test protection against memory exhaustion attacks
        
        // Try to create metadata claiming huge string length
        let mut bad_data = Vec::new();
        bad_data.extend_from_slice(&8u64.to_le_bytes()); // key length
        bad_data.extend_from_slice(b"test_key");
        bad_data.extend_from_slice(&8u32.to_le_bytes()); // string type
        bad_data.extend_from_slice(&(u64::MAX).to_le_bytes()); // claim enormous string
        bad_data.extend_from_slice(b"tiny"); // but provide tiny data
        
        let mut cursor = Cursor::new(&bad_data);
        let result = Metadata::read(&mut cursor, 1);
        
        // Should fail gracefully, not try to allocate huge amounts of memory
        assert!(result.is_err());
    }
}