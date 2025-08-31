//! End-to-end integration tests

use gguf::prelude::*;
use std::collections::HashMap;
use std::io::Cursor;
use tempfile::NamedTempFile;
use std::io::Write as IoWrite;

#[test]
fn test_complete_workflow() {
    // Create a complex model with multiple tensors and metadata
    let mut builder = GGUFBuilder::simple("integration_test_model", "A comprehensive test model");
    
    // Add rich metadata
    builder = builder
        .add_metadata("model.version", MetadataValue::String("1.0.0".to_string()))
        .add_metadata("model.author", MetadataValue::String("GGUF Test Suite".to_string()))
        .add_metadata("model.license", MetadataValue::String("MIT".to_string()))
        .add_metadata("model.layers", MetadataValue::UInt32(12))
        .add_metadata("model.parameters", MetadataValue::UInt64(175_000_000))
        .add_metadata("model.vocab_size", MetadataValue::UInt32(50257))
        .add_metadata("model.context_length", MetadataValue::UInt32(2048))
        .add_metadata("model.embedding_length", MetadataValue::UInt32(768))
        .add_metadata("model.feed_forward_length", MetadataValue::UInt32(3072))
        .add_metadata("model.attention_heads", MetadataValue::UInt32(12))
        .add_metadata("model.fine_tuned", MetadataValue::Bool(false))
        .add_metadata("training.learning_rate", MetadataValue::F32(0.0001))
        .add_metadata("training.batch_size", MetadataValue::UInt32(32))
        .add_metadata("training.epochs", MetadataValue::UInt32(10));

    // Add embedding matrix
    let vocab_size = 50257;
    let embed_dim = 768;
    let embedding_weights: Vec<f32> = (0..(vocab_size * embed_dim))
        .map(|i| (i as f32 * 0.001) % 1.0) // Simple pattern for testing
        .collect();
    
    builder = builder.add_f32_tensor(
        "token_embd.weight", 
        vec![vocab_size as u64, embed_dim as u64], 
        embedding_weights.clone()
    );

    // Add transformer layers
    for layer in 0..12 {
        // Attention weights
        let attn_weights: Vec<f32> = (0..(embed_dim * embed_dim))
            .map(|i| ((i + layer * 1000) as f32 * 0.0001) % 0.1)
            .collect();
        
        builder = builder.add_f32_tensor(
            &format!("blk.{}.attn_q.weight", layer),
            vec![embed_dim as u64, embed_dim as u64],
            attn_weights.clone()
        );
        
        builder = builder.add_f32_tensor(
            &format!("blk.{}.attn_k.weight", layer),
            vec![embed_dim as u64, embed_dim as u64],
            attn_weights.clone()
        );
        
        builder = builder.add_f32_tensor(
            &format!("blk.{}.attn_v.weight", layer),
            vec![embed_dim as u64, embed_dim as u64],
            attn_weights.clone()
        );
        
        builder = builder.add_f32_tensor(
            &format!("blk.{}.attn_output.weight", layer),
            vec![embed_dim as u64, embed_dim as u64],
            attn_weights
        );
        
        // Feed-forward weights
        let ff_dim = 3072;
        let ff_up_weights: Vec<f32> = (0..(embed_dim * ff_dim))
            .map(|i| ((i + layer * 2000) as f32 * 0.0001) % 0.1)
            .collect();
        
        let ff_down_weights: Vec<f32> = (0..(ff_dim * embed_dim))
            .map(|i| ((i + layer * 3000) as f32 * 0.0001) % 0.1)
            .collect();
        
        builder = builder
            .add_f32_tensor(
                &format!("blk.{}.ffn_up.weight", layer),
                vec![embed_dim as u64, ff_dim as u64],
                ff_up_weights
            )
            .add_f32_tensor(
                &format!("blk.{}.ffn_down.weight", layer),
                vec![ff_dim as u64, embed_dim as u64],
                ff_down_weights
            );
        
        // Layer normalization
        let ln_weights: Vec<f32> = (0..embed_dim)
            .map(|i| 1.0 + (i as f32 * 0.001) % 0.1)
            .collect();
        
        let ln_bias: Vec<f32> = (0..embed_dim)
            .map(|i| (i as f32 * 0.0001) % 0.01)
            .collect();
        
        builder = builder
            .add_f32_tensor(
                &format!("blk.{}.attn_norm.weight", layer),
                vec![embed_dim as u64],
                ln_weights.clone()
            )
            .add_f32_tensor(
                &format!("blk.{}.attn_norm.bias", layer),
                vec![embed_dim as u64],
                ln_bias.clone()
            )
            .add_f32_tensor(
                &format!("blk.{}.ffn_norm.weight", layer),
                vec![embed_dim as u64],
                ln_weights
            )
            .add_f32_tensor(
                &format!("blk.{}.ffn_norm.bias", layer),
                vec![embed_dim as u64],
                ln_bias
            );
    }
    
    // Add final layer norm and output projection
    let final_ln: Vec<f32> = (0..embed_dim).map(|i| 1.0 + (i as f32 * 0.001) % 0.1).collect();
    let final_ln_bias: Vec<f32> = (0..embed_dim).map(|i| (i as f32 * 0.0001) % 0.01).collect();
    
    builder = builder
        .add_f32_tensor("output_norm.weight", vec![embed_dim as u64], final_ln)
        .add_f32_tensor("output_norm.bias", vec![embed_dim as u64], final_ln_bias)
        .add_f32_tensor(
            "output.weight",
            vec![embed_dim as u64, vocab_size as u64],
            embedding_weights // Reuse embedding weights for output projection
        );

    // Build the model
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let result = builder.build_to_file(temp_file.path()).expect("Failed to build model");
    
    println!("Built model with {} bytes, {} tensors", result.total_bytes_written, result.tensor_count);
    assert!(result.total_bytes_written > 1_000_000); // Should be substantial
    assert_eq!(result.tensor_count, 1 + 12 * 8 + 2); // embedding + 12 layers * 8 tensors + final norm + output

    // Read back and verify
    let reader = gguf::reader::open_gguf_file(temp_file.path()).expect("Failed to read model");
    
    // Verify metadata
    assert_eq!(reader.metadata().get_string("general.name"), Some("integration_test_model"));
    assert_eq!(reader.metadata().get_u32("model.layers"), Some(12));
    assert_eq!(reader.metadata().get_u64("model.parameters"), Some(175_000_000));
    assert_eq!(reader.metadata().get_bool("model.fine_tuned"), Some(false));
    assert_eq!(reader.metadata().get_f32("training.learning_rate"), Some(0.0001));
    
    // Verify tensors exist and have correct shapes
    assert!(reader.get_tensor_info("token_embd.weight").is_some());
    assert!(reader.get_tensor_info("blk.0.attn_q.weight").is_some());
    assert!(reader.get_tensor_info("blk.11.ffn_down.weight").is_some());
    assert!(reader.get_tensor_info("output.weight").is_some());
    
    let embedding_info = reader.get_tensor_info("token_embd.weight").unwrap();
    assert_eq!(embedding_info.shape().dimensions(), &[vocab_size as u64, embed_dim as u64]);
    assert_eq!(embedding_info.tensor_type(), TensorType::F32);
    
    // Load and verify some tensor data
    let mut mutable_reader = reader;
    let embedding_data = mutable_reader.load_tensor_data("token_embd.weight")
        .expect("Failed to load embedding data")
        .expect("Embedding data should exist");
    
    assert_eq!(embedding_data.len(), vocab_size * embed_dim * 4); // f32 = 4 bytes
    
    // Verify first few values match what we put in
    let floats: Vec<f32> = embedding_data.as_slice()
        .chunks_exact(4)
        .take(10)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    
    let expected: Vec<f32> = (0..10).map(|i| (i as f32 * 0.001) % 1.0).collect();
    for (actual, expected) in floats.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < f32::EPSILON);
    }
}

#[test]
fn test_round_trip_data_integrity() {
    // Test that data survives round-trip encoding/decoding perfectly
    
    let test_cases = vec![
        // (name, shape, data)
        ("zeros", vec![100], vec![0.0f32; 100]),
        ("ones", vec![50], vec![1.0f32; 50]),
        ("sequence", vec![20], (0..20).map(|i| i as f32).collect()),
        ("random_pattern", vec![10, 10], (0..100).map(|i| ((i * 17) % 97) as f32 / 97.0).collect()),
        ("negative_values", vec![30], (0..30).map(|i| -((i as f32) - 15.0)).collect()),
        ("large_values", vec![25], (0..25).map(|i| (i as f32) * 1000000.0).collect()),
        ("small_values", vec![25], (0..25).map(|i| (i as f32) * 0.000001).collect()),
    ];
    
    for (name, shape, original_data) in test_cases {
        let mut builder = GGUFBuilder::new();
        builder = builder.add_f32_tensor(name, shape.clone(), original_data.clone());
        
        let bytes = builder.build_to_bytes().expect("Failed to build");
        
        let cursor = Cursor::new(bytes);
        let mut reader = GGUFFileReader::new(cursor).expect("Failed to read");
        
        let loaded_data = reader.load_tensor_data(name)
            .expect("Failed to load tensor")
            .expect("Tensor should exist");
        
        // Convert back to f32 values
        let loaded_floats: Vec<f32> = loaded_data.as_slice()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        assert_eq!(loaded_floats.len(), original_data.len());
        
        for (i, (&original, &loaded)) in original_data.iter().zip(loaded_floats.iter()).enumerate() {
            assert!(
                (original - loaded).abs() < f32::EPSILON,
                "Mismatch at index {} for tensor '{}': expected {}, got {}",
                i, name, original, loaded
            );
        }
    }
}

#[test]
fn test_mixed_tensor_types() {
    // Test a model with multiple tensor types
    
    let mut builder = GGUFBuilder::simple("mixed_types_model", "Model with various tensor types");
    
    // F32 tensor
    let f32_data = vec![1.0f32, 2.0, 3.0, 4.0];
    builder = builder.add_f32_tensor("f32_tensor", vec![2, 2], f32_data.clone());
    
    // I32 tensor  
    let i32_data = vec![10i32, -20, 30, -40, 50];
    builder = builder.add_i32_tensor("i32_tensor", vec![5], i32_data.clone());
    
    // Add some quantized data (simulated)
    let q4_data = vec![0u8; 36]; // 2 blocks of Q4_0 data (18 bytes each)
    builder = builder.add_quantized_tensor("q4_tensor", vec![64], TensorType::Q4_0, q4_data);
    
    let q8_data = vec![0u8; 68]; // 2 blocks of Q8_0 data (34 bytes each)  
    builder = builder.add_quantized_tensor("q8_tensor", vec![64], TensorType::Q8_0, q8_data);
    
    // Build and read back
    let bytes = builder.build_to_bytes().expect("Failed to build mixed model");
    
    let cursor = Cursor::new(bytes);
    let mut reader = GGUFFileReader::new(cursor).expect("Failed to read mixed model");
    
    // Verify all tensors exist with correct types
    let f32_info = reader.get_tensor_info("f32_tensor").unwrap();
    assert_eq!(f32_info.tensor_type(), TensorType::F32);
    assert_eq!(f32_info.shape().dimensions(), &[2, 2]);
    
    let i32_info = reader.get_tensor_info("i32_tensor").unwrap();
    assert_eq!(i32_info.tensor_type(), TensorType::I32);
    assert_eq!(i32_info.shape().dimensions(), &[5]);
    
    let q4_info = reader.get_tensor_info("q4_tensor").unwrap();
    assert_eq!(q4_info.tensor_type(), TensorType::Q4_0);
    assert_eq!(q4_info.shape().dimensions(), &[64]);
    
    let q8_info = reader.get_tensor_info("q8_tensor").unwrap();
    assert_eq!(q8_info.tensor_type(), TensorType::Q8_0);
    assert_eq!(q8_info.shape().dimensions(), &[64]);
    
    // Load and verify F32 data
    let loaded_f32 = reader.load_tensor_data("f32_tensor")
        .expect("Failed to load f32 tensor")
        .expect("F32 tensor should exist");
    
    let f32_values: Vec<f32> = loaded_f32.as_slice()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    
    assert_eq!(f32_values, f32_data);
    
    // Load and verify I32 data
    let loaded_i32 = reader.load_tensor_data("i32_tensor")
        .expect("Failed to load i32 tensor")
        .expect("I32 tensor should exist");
    
    let i32_values: Vec<i32> = loaded_i32.as_slice()
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    
    assert_eq!(i32_values, i32_data);
    
    // Verify quantized data can be loaded (even if we can't verify content)
    let loaded_q4 = reader.load_tensor_data("q4_tensor")
        .expect("Failed to load q4 tensor")
        .expect("Q4 tensor should exist");
    assert_eq!(loaded_q4.len(), 36);
    
    let loaded_q8 = reader.load_tensor_data("q8_tensor")
        .expect("Failed to load q8 tensor")
        .expect("Q8 tensor should exist");
    assert_eq!(loaded_q8.len(), 68);
}

#[test] 
fn test_metadata_round_trip() {
    // Test comprehensive metadata round-trip
    
    let mut metadata_builder = MetadataBuilder::new()
        .add_string("model.name", "comprehensive_test")
        .add_string("model.description", "Testing all metadata types")
        .add_u8("uint8_val", 255)
        .add_i8("int8_val", -128)
        .add_u16("uint16_val", 65535)  
        .add_i16("int16_val", -32768)
        .add_u32("uint32_val", 4294967295)
        .add_i32("int32_val", -2147483648)
        .add_u64("uint64_val", 18446744073709551615u64)
        .add_i64("int64_val", -9223372036854775808i64)
        .add_f32("float32_val", std::f32::consts::PI)
        .add_f64("float64_val", std::f64::consts::E)
        .add_bool("bool_true", true)
        .add_bool("bool_false", false);
    
    // Add array values
    let array_values = vec![
        MetadataValue::UInt32(1),
        MetadataValue::UInt32(2), 
        MetadataValue::UInt32(3),
        MetadataValue::UInt32(4),
        MetadataValue::UInt32(5),
    ];
    metadata_builder = metadata_builder.add_array("array_val", array_values);
    
    let metadata = metadata_builder.build();
    
    let builder = GGUFBuilder::with_metadata(metadata.clone());
    let bytes = builder.build_to_bytes().expect("Failed to build");
    
    let cursor = Cursor::new(bytes);
    let reader = GGUFFileReader::new(cursor).expect("Failed to read");
    
    let loaded_metadata = reader.metadata();
    
    // Verify all values
    assert_eq!(loaded_metadata.get_string("model.name"), Some("comprehensive_test"));
    assert_eq!(loaded_metadata.get_string("model.description"), Some("Testing all metadata types"));
    
    assert_eq!(loaded_metadata.get("uint8_val").and_then(|v| v.as_u8()), Some(255));
    assert_eq!(loaded_metadata.get("int8_val").and_then(|v| v.as_i8()), Some(-128));
    assert_eq!(loaded_metadata.get("uint16_val").and_then(|v| v.as_u16()), Some(65535));
    assert_eq!(loaded_metadata.get("int16_val").and_then(|v| v.as_i16()), Some(-32768));
    assert_eq!(loaded_metadata.get("uint32_val").and_then(|v| v.as_u32()), Some(4294967295));
    assert_eq!(loaded_metadata.get("int32_val").and_then(|v| v.as_i32()), Some(-2147483648));
    assert_eq!(loaded_metadata.get("uint64_val").and_then(|v| v.as_u64()), Some(18446744073709551615));
    assert_eq!(loaded_metadata.get("int64_val").and_then(|v| v.as_i64()), Some(-9223372036854775808));
    
    let loaded_f32 = loaded_metadata.get("float32_val").and_then(|v| v.as_f32()).unwrap();
    assert!((loaded_f32 - std::f32::consts::PI).abs() < f32::EPSILON);
    
    let loaded_f64 = loaded_metadata.get("float64_val").and_then(|v| v.as_f64()).unwrap();
    assert!((loaded_f64 - std::f64::consts::E).abs() < f64::EPSILON);
    
    assert_eq!(loaded_metadata.get("bool_true").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(loaded_metadata.get("bool_false").and_then(|v| v.as_bool()), Some(false));
    
    // Verify array
    if let Some(MetadataValue::Array(arr)) = loaded_metadata.get("array_val") {
        assert_eq!(arr.len(), 5);
        for (i, value) in arr.iter().enumerate() {
            assert_eq!(value.as_u32(), Some((i + 1) as u32));
        }
    } else {
        panic!("Expected array value");
    }
}