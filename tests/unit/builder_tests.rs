//! Unit tests for the builder module

use gguf::builder::*;
use gguf::prelude::*;
use gguf::reader::GGUFFileReader;
use std::io::Cursor;
use tempfile::NamedTempFile;
use std::io::Write as IoWrite;

mod gguf_builder_tests {
    use super::*;

    #[test]
    fn test_gguf_builder_creation() {
        let builder = GGUFBuilder::new();
        
        assert_eq!(builder.tensor_count(), 0);
        assert_eq!(builder.metadata_count(), 0);
    }

    #[test]
    fn test_gguf_builder_simple() {
        let builder = GGUFBuilder::simple("test_model", "A test model");
        
        assert_eq!(builder.tensor_count(), 0);
        assert_eq!(builder.metadata_count(), 2); // name and description
    }

    #[test]
    fn test_gguf_builder_with_metadata() {
        let mut metadata = Metadata::new();
        metadata.insert("custom_key".to_string(), MetadataValue::UInt32(42));
        
        let builder = GGUFBuilder::with_metadata(metadata);
        
        assert_eq!(builder.tensor_count(), 0);
        assert_eq!(builder.metadata_count(), 1);
    }

    #[test]
    fn test_gguf_builder_add_metadata() {
        let mut builder = GGUFBuilder::new();
        
        builder = builder.add_metadata("key1", MetadataValue::String("value1".to_string()));
        builder = builder.add_metadata("key2", MetadataValue::UInt64(12345));
        
        assert_eq!(builder.metadata_count(), 2);
    }

    #[test]
    fn test_gguf_builder_add_tensor() {
        let mut builder = GGUFBuilder::new();
        
        let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        builder = builder.add_tensor(
            "weights", 
            vec![2, 2], 
            TensorType::F32, 
            data.clone()
        ).expect("Failed to add tensor");
        
        assert_eq!(builder.tensor_count(), 1);
    }

    #[test]
    fn test_gguf_builder_add_tensor_with_shape() {
        let mut builder = GGUFBuilder::new();
        
        let shape = TensorShape::new(vec![3, 3]);
        let data = TensorData::F32(vec![0.0f32; 9]);
        
        builder = builder.add_tensor_with_shape("matrix", shape, TensorType::F32, data);
        
        assert_eq!(builder.tensor_count(), 1);
    }

    #[test]
    fn test_gguf_builder_add_f32_tensor() {
        let mut builder = GGUFBuilder::new();
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        builder = builder.add_f32_tensor("weights", vec![2, 3], data);
        
        assert_eq!(builder.tensor_count(), 1);
    }

    #[test]
    fn test_gguf_builder_add_i32_tensor() {
        let mut builder = GGUFBuilder::new();
        
        let data = vec![1, 2, 3, 4];
        builder = builder.add_i32_tensor("indices", vec![4], data);
        
        assert_eq!(builder.tensor_count(), 1);
    }

    #[test]
    fn test_gguf_builder_add_quantized_tensor() {
        let mut builder = GGUFBuilder::new();
        
        // Create mock quantized data (64 bytes for 128 elements in Q4_0)
        let quantized_data = vec![0u8; 72]; // 4 blocks * 18 bytes per block
        
        builder = builder.add_quantized_tensor(
            "quantized_weights",
            vec![128], // 128 elements = 4 blocks of 32 elements each
            TensorType::Q4_0,
            quantized_data
        );
        
        assert_eq!(builder.tensor_count(), 1);
    }

    #[test]
    fn test_gguf_builder_build_to_bytes() {
        let mut builder = GGUFBuilder::simple("test_model", "Test model");
        
        let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        builder = builder.add_tensor(
            "weights", 
            vec![2, 2], 
            TensorType::F32, 
            data
        ).expect("Failed to add tensor");
        
        let bytes = builder.build_to_bytes().expect("Failed to build to bytes");
        
        assert!(!bytes.is_empty());
        assert!(bytes.len() > 100); // Should be substantial size
        
        // Verify we can read it back
        let cursor = Cursor::new(bytes);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read built data");
        
        assert_eq!(reader.tensor_count(), 1);
        assert_eq!(reader.metadata().len(), 2); // model name and description
        assert!(reader.get_tensor_info("weights").is_some());
    }

    #[test]
    fn test_gguf_builder_build_to_file() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let file_path = temp_file.path();
        
        let mut builder = GGUFBuilder::simple("file_model", "Model saved to file");
        
        let data = vec![10i32, 20i32, 30i32];
        builder = builder.add_i32_tensor("data", vec![3], data);
        
        let result = builder.build_to_file(file_path).expect("Failed to build to file");
        
        assert!(result.total_bytes_written > 0);
        assert_eq!(result.tensor_count, 1);
        assert_eq!(result.metadata_count, 2);
        
        // Verify file was created and can be read
        use gguf::reader::open_gguf_file;
        let reader = open_gguf_file(file_path).expect("Failed to read file");
        
        assert_eq!(reader.tensor_count(), 1);
        assert_eq!(reader.metadata().get_string("general.name"), Some("file_model"));
        assert_eq!(reader.metadata().get_string("general.description"), Some("Model saved to file"));
    }

    #[test]
    fn test_gguf_builder_build_to_writer() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut builder = GGUFBuilder::new();
        builder = builder.add_metadata("test_key", MetadataValue::Bool(true));
        builder = builder.add_f32_tensor("test_tensor", vec![1], vec![42.0]);
        
        let result = builder.build_to_writer(cursor).expect("Failed to build to writer");
        
        assert!(result.total_bytes_written > 0);
        assert_eq!(result.tensor_count, 1);
        assert_eq!(result.metadata_count, 1);
        
        // Verify the data
        let cursor = Cursor::new(&buffer);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read data");
        
        assert_eq!(reader.tensor_count(), 1);
        assert_eq!(reader.metadata().get_bool("test_key"), Some(true));
    }

    #[test]
    fn test_gguf_builder_complex_model() {
        let mut builder = GGUFBuilder::simple("complex_model", "A model with multiple tensors");
        
        // Add various metadata
        builder = builder
            .add_metadata("model.version", MetadataValue::String("1.0".to_string()))
            .add_metadata("model.layers", MetadataValue::UInt32(12))
            .add_metadata("model.parameters", MetadataValue::UInt64(1000000))
            .add_metadata("model.quantized", MetadataValue::Bool(true));
        
        // Add multiple tensors of different types
        builder = builder
            .add_f32_tensor("embedding.weight", vec![50000, 768], vec![0.0f32; 50000 * 768])
            .add_f32_tensor("layer.0.attention.weight", vec![768, 768], vec![1.0f32; 768 * 768])
            .add_i32_tensor("tokenizer.vocab", vec![50000], (0..50000).collect());
        
        // Add quantized tensor
        let quantized_data = vec![0u8; 1000]; // Mock quantized data
        builder = builder.add_quantized_tensor(
            "layer.0.mlp.weight.q4_0",
            vec![768, 3072],
            TensorType::Q4_0,
            quantized_data
        );
        
        let bytes = builder.build_to_bytes().expect("Failed to build complex model");
        
        // Verify the built model
        let cursor = Cursor::new(bytes);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read complex model");
        
        assert_eq!(reader.tensor_count(), 4);
        assert_eq!(reader.metadata().len(), 6); // 2 from simple() + 4 added
        
        // Verify metadata
        assert_eq!(reader.metadata().get_string("general.name"), Some("complex_model"));
        assert_eq!(reader.metadata().get_string("model.version"), Some("1.0"));
        assert_eq!(reader.metadata().get_u32("model.layers"), Some(12));
        assert_eq!(reader.metadata().get_bool("model.quantized"), Some(true));
        
        // Verify tensors
        assert!(reader.get_tensor_info("embedding.weight").is_some());
        assert!(reader.get_tensor_info("layer.0.attention.weight").is_some());
        assert!(reader.get_tensor_info("tokenizer.vocab").is_some());
        assert!(reader.get_tensor_info("layer.0.mlp.weight.q4_0").is_some());
        
        let embedding_info = reader.get_tensor_info("embedding.weight").unwrap();
        assert_eq!(embedding_info.shape().dimensions(), &[50000, 768]);
        assert_eq!(embedding_info.tensor_type(), TensorType::F32);
        
        let quantized_info = reader.get_tensor_info("layer.0.mlp.weight.q4_0").unwrap();
        assert_eq!(quantized_info.tensor_type(), TensorType::Q4_0);
    }

    #[test]
    fn test_gguf_builder_error_handling() {
        let mut builder = GGUFBuilder::new();
        
        // Test adding tensor with mismatched data size
        let data = vec![1.0f32, 2.0f32]; // 2 elements
        let result = builder.add_tensor(
            "mismatched", 
            vec![3], // But shape says 3 elements
            TensorType::F32, 
            data
        );
        
        // Should return an error
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_builder_empty_model() {
        let builder = GGUFBuilder::new();
        
        let bytes = builder.build_to_bytes().expect("Failed to build empty model");
        
        let cursor = Cursor::new(bytes);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read empty model");
        
        assert_eq!(reader.tensor_count(), 0);
        assert_eq!(reader.metadata().len(), 0);
    }

    #[test]
    fn test_gguf_builder_duplicate_tensor_names() {
        let mut builder = GGUFBuilder::new();
        
        builder = builder.add_f32_tensor("tensor", vec![2], vec![1.0, 2.0]);
        
        // Adding another tensor with the same name should replace the first
        builder = builder.add_f32_tensor("tensor", vec![3], vec![3.0, 4.0, 5.0]);
        
        assert_eq!(builder.tensor_count(), 1); // Should still be 1
        
        let bytes = builder.build_to_bytes().expect("Failed to build");
        let cursor = Cursor::new(bytes);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read");
        
        let tensor_info = reader.get_tensor_info("tensor").unwrap();
        assert_eq!(tensor_info.shape().dimensions(), &[3]); // Should be the second tensor
    }

    #[test]
    fn test_gguf_builder_large_metadata() {
        let mut builder = GGUFBuilder::new();
        
        // Add many metadata entries
        for i in 0..1000 {
            builder = builder.add_metadata(
                &format!("key_{}", i),
                MetadataValue::UInt32(i as u32)
            );
        }
        
        assert_eq!(builder.metadata_count(), 1000);
        
        let bytes = builder.build_to_bytes().expect("Failed to build with large metadata");
        
        let cursor = Cursor::new(bytes);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read large metadata");
        
        assert_eq!(reader.metadata().len(), 1000);
        assert_eq!(reader.metadata().get_u32("key_500"), Some(500));
    }
}

mod metadata_builder_tests {
    use super::*;

    #[test]
    fn test_metadata_builder_creation() {
        let builder = MetadataBuilder::new();
        
        assert_eq!(builder.len(), 0);
        assert!(builder.is_empty());
    }

    #[test]
    fn test_metadata_builder_add() {
        let mut builder = MetadataBuilder::new();
        
        builder = builder.add("string_key", MetadataValue::String("test".to_string()));
        builder = builder.add("int_key", MetadataValue::I32(-42));
        builder = builder.add("float_key", MetadataValue::F32(3.14));
        
        assert_eq!(builder.len(), 3);
        assert!(!builder.is_empty());
    }

    #[test]
    fn test_metadata_builder_convenience_methods() {
        let mut builder = MetadataBuilder::new();
        
        builder = builder
            .add_string("name", "test_model")
            .add_u32("version", 1)
            .add_i32("layers", 12)
            .add_u64("parameters", 175_000_000)
            .add_i64("timestamp", -1234567890)
            .add_f32("learning_rate", 0.001)
            .add_f64("accuracy", 0.95123456789)
            .add_bool("fine_tuned", true);
        
        assert_eq!(builder.len(), 8);
        
        let metadata = builder.build();
        
        assert_eq!(metadata.get_string("name"), Some("test_model"));
        assert_eq!(metadata.get_u32("version"), Some(1));
        assert_eq!(metadata.get_i32("layers"), Some(12));
        assert_eq!(metadata.get_u64("parameters"), Some(175_000_000));
        assert_eq!(metadata.get_i64("timestamp"), Some(-1234567890));
        assert_eq!(metadata.get_f32("learning_rate"), Some(0.001));
        assert_eq!(metadata.get_f64("accuracy"), Some(0.95123456789));
        assert_eq!(metadata.get_bool("fine_tuned"), Some(true));
    }

    #[test]
    fn test_metadata_builder_add_array() {
        let mut builder = MetadataBuilder::new();
        
        let array_values = vec![
            MetadataValue::UInt32(1),
            MetadataValue::UInt32(2),
            MetadataValue::UInt32(3),
        ];
        
        builder = builder.add_array("numbers", array_values.clone());
        
        assert_eq!(builder.len(), 1);
        
        let metadata = builder.build();
        
        if let Some(MetadataValue::Array(arr)) = metadata.get("numbers") {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0].as_u32(), Some(1));
            assert_eq!(arr[1].as_u32(), Some(2));
            assert_eq!(arr[2].as_u32(), Some(3));
        } else {
            panic!("Expected array value");
        }
    }

    #[test]
    fn test_metadata_builder_model_info() {
        let builder = MetadataBuilder::model_info(
            "GPT-2",
            "OpenAI GPT-2 model",
            "gpt2",
            "transformers"
        );
        
        assert_eq!(builder.len(), 4);
        
        let metadata = builder.build();
        
        assert_eq!(metadata.get_string("general.name"), Some("GPT-2"));
        assert_eq!(metadata.get_string("general.description"), Some("OpenAI GPT-2 model"));
        assert_eq!(metadata.get_string("general.architecture"), Some("gpt2"));
        assert_eq!(metadata.get_string("general.file_type"), Some("transformers"));
    }

    #[test]
    fn test_metadata_builder_tokenizer_info() {
        let builder = MetadataBuilder::tokenizer_info(
            50257,
            Some(50256),
            Some(50256),
            Some(50257)
        );
        
        assert_eq!(builder.len(), 4);
        
        let metadata = builder.build();
        
        assert_eq!(metadata.get_u32("tokenizer.ggml.vocab_size"), Some(50257));
        assert_eq!(metadata.get_u32("tokenizer.ggml.bos_token_id"), Some(50256));
        assert_eq!(metadata.get_u32("tokenizer.ggml.eos_token_id"), Some(50256));
        assert_eq!(metadata.get_u32("tokenizer.ggml.pad_token_id"), Some(50257));
    }

    #[test]
    fn test_metadata_builder_chain_methods() {
        let metadata = MetadataBuilder::new()
            .add_string("step1", "first")
            .add_string("step2", "second")
            .add_string("step3", "third")
            .build();
        
        assert_eq!(metadata.len(), 3);
        assert_eq!(metadata.get_string("step1"), Some("first"));
        assert_eq!(metadata.get_string("step2"), Some("second"));
        assert_eq!(metadata.get_string("step3"), Some("third"));
    }

    #[test]
    fn test_metadata_builder_overwrite() {
        let mut builder = MetadataBuilder::new();
        
        builder = builder.add_string("key", "first_value");
        builder = builder.add_string("key", "second_value"); // Should overwrite
        
        assert_eq!(builder.len(), 1); // Still only one key
        
        let metadata = builder.build();
        assert_eq!(metadata.get_string("key"), Some("second_value"));
    }

    #[test]
    fn test_metadata_builder_from_existing() {
        let mut existing = Metadata::new();
        existing.insert("existing_key".to_string(), MetadataValue::UInt32(100));
        
        let builder = MetadataBuilder::from_metadata(existing)
            .add_string("new_key", "new_value");
        
        assert_eq!(builder.len(), 2);
        
        let metadata = builder.build();
        assert_eq!(metadata.get_u32("existing_key"), Some(100));
        assert_eq!(metadata.get_string("new_key"), Some("new_value"));
    }

    #[test]
    fn test_metadata_builder_clear() {
        let mut builder = MetadataBuilder::new()
            .add_string("key1", "value1")
            .add_string("key2", "value2");
        
        assert_eq!(builder.len(), 2);
        
        builder = builder.clear();
        
        assert_eq!(builder.len(), 0);
        assert!(builder.is_empty());
    }
}

mod tensor_builder_tests {
    use super::*;

    #[test]
    fn test_tensor_builder_creation() {
        let builder = TensorBuilder::new("test_tensor", TensorType::F32);
        
        assert_eq!(builder.name(), "test_tensor");
        assert_eq!(builder.tensor_type(), TensorType::F32);
    }

    #[test]
    fn test_tensor_builder_with_shape() {
        let shape = TensorShape::new(vec![10, 20]);
        let builder = TensorBuilder::new("tensor", TensorType::F32)
            .with_shape(shape.clone());
        
        assert_eq!(builder.shape(), &shape);
    }

    #[test]
    fn test_tensor_builder_with_dimensions() {
        let builder = TensorBuilder::new("tensor", TensorType::I32)
            .with_dimensions(vec![5, 5, 5]);
        
        assert_eq!(builder.shape().dimensions(), &[5, 5, 5]);
        assert_eq!(builder.shape().element_count(), 125);
    }

    #[test]
    fn test_tensor_builder_with_data() {
        let data = TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]);
        let builder = TensorBuilder::new("tensor", TensorType::F32)
            .with_dimensions(vec![2, 2])
            .with_data(data.clone());
        
        // Note: This test assumes the builder has methods to access the data
        // The actual API might be different
    }

    #[test]
    fn test_tensor_builder_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let builder = TensorBuilder::f32("weights", vec![2, 3], data.clone());
        
        assert_eq!(builder.name(), "weights");
        assert_eq!(builder.tensor_type(), TensorType::F32);
        assert_eq!(builder.shape().dimensions(), &[2, 3]);
    }

    #[test]
    fn test_tensor_builder_i32() {
        let data = vec![1i32, 2, 3, 4];
        let builder = TensorBuilder::i32("indices", vec![4], data.clone());
        
        assert_eq!(builder.name(), "indices");
        assert_eq!(builder.tensor_type(), TensorType::I32);
        assert_eq!(builder.shape().dimensions(), &[4]);
    }

    #[test]
    fn test_tensor_builder_quantized() {
        let quantized_data = vec![0u8; 72]; // Mock quantized data
        let builder = TensorBuilder::quantized(
            "q_weights",
            vec![128], // 128 elements
            TensorType::Q4_0,
            quantized_data
        );
        
        assert_eq!(builder.name(), "q_weights");
        assert_eq!(builder.tensor_type(), TensorType::Q4_0);
        assert_eq!(builder.shape().dimensions(), &[128]);
    }

    #[test]
    fn test_tensor_builder_build() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let builder = TensorBuilder::f32("test", vec![2, 2], data);
        
        let (info, tensor_data) = builder.build();
        
        assert_eq!(info.name(), "test");
        assert_eq!(info.tensor_type(), TensorType::F32);
        assert_eq!(info.shape().dimensions(), &[2, 2]);
        assert_eq!(info.element_count(), 4);
        
        assert_eq!(tensor_data.len(), 16); // 4 * 4 bytes
    }

    #[test]
    fn test_tensor_builder_validation() {
        // Test that mismatched data size is caught
        let data = vec![1.0f32, 2.0]; // 2 elements
        let result = TensorBuilder::f32("invalid", vec![3], data).validate();
        
        // Should return an error for mismatched sizes
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_builder_empty_tensor() {
        let data: Vec<f32> = vec![];
        let builder = TensorBuilder::f32("empty", vec![0], data);
        
        let (info, tensor_data) = builder.build();
        
        assert_eq!(info.element_count(), 0);
        assert_eq!(tensor_data.len(), 0);
    }

    #[test]
    fn test_tensor_builder_large_tensor() {
        let size = 1000 * 1000; // 1M elements
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let builder = TensorBuilder::f32("large", vec![1000, 1000], data);
        
        let (info, tensor_data) = builder.build();
        
        assert_eq!(info.element_count(), size as u64);
        assert_eq!(tensor_data.len(), size * 4); // f32 = 4 bytes each
    }
}