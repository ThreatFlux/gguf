//! Property-based tests for tensor operations

use gguf::prelude::*;
use proptest::prelude::*;
use std::io::Cursor;

// Strategy for generating valid tensor shapes
fn tensor_shape_strategy() -> impl Strategy<Value = Vec<u64>> {
    prop::collection::vec(1u64..1000, 1..5) // 1-5 dimensions, each 1-1000 elements
}

// Strategy for generating tensor names
fn tensor_name_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z][a-zA-Z0-9._-]{0,50}" // Valid tensor name pattern
}

// Strategy for generating f32 data
fn f32_data_strategy(size: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1000.0f32..1000.0f32, size..=size)
}

// Strategy for generating i32 data  
fn i32_data_strategy(size: usize) -> impl Strategy<Value = Vec<i32>> {
    prop::collection::vec(-1000i32..1000i32, size..=size)
}

proptest! {
    #[test]
    fn test_tensor_round_trip_f32(
        name in tensor_name_strategy(),
        shape in tensor_shape_strategy(),
        data in f32_data_strategy(1..10000).prop_flat_map(|data| Just(data))
    ) {
        // Skip if data size doesn't match shape
        let expected_elements = shape.iter().product::<u64>() as usize;
        prop_assume!(data.len() == expected_elements);
        
        // Build tensor
        let mut builder = GGUFBuilder::new();
        builder = builder.add_f32_tensor(&name, shape.clone(), data.clone());
        
        let bytes = builder.build_to_bytes().expect("Failed to build");
        
        // Read back
        let cursor = Cursor::new(bytes);
        let mut reader = GGUFFileReader::new(cursor).expect("Failed to read");
        
        // Verify tensor info
        let tensor_info = reader.get_tensor_info(&name).unwrap();
        prop_assert_eq!(tensor_info.name(), name);
        prop_assert_eq!(tensor_info.tensor_type(), TensorType::F32);
        prop_assert_eq!(tensor_info.shape().dimensions(), shape.as_slice());
        prop_assert_eq!(tensor_info.element_count() as usize, expected_elements);
        
        // Load and verify data
        let loaded_data = reader.load_tensor_data(&name)
            .expect("Failed to load")
            .expect("Should exist");
        
        let loaded_floats: Vec<f32> = loaded_data.as_slice()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        prop_assert_eq!(loaded_floats.len(), data.len());
        
        for (original, loaded) in data.iter().zip(loaded_floats.iter()) {
            prop_assert!((original - loaded).abs() < f32::EPSILON);
        }
    }
    
    #[test]
    fn test_tensor_round_trip_i32(
        name in tensor_name_strategy(),
        shape in tensor_shape_strategy(),
        data in i32_data_strategy(1..10000).prop_flat_map(|data| Just(data))
    ) {
        let expected_elements = shape.iter().product::<u64>() as usize;
        prop_assume!(data.len() == expected_elements);
        
        let mut builder = GGUFBuilder::new();
        builder = builder.add_i32_tensor(&name, shape.clone(), data.clone());
        
        let bytes = builder.build_to_bytes().expect("Failed to build");
        
        let cursor = Cursor::new(bytes);
        let mut reader = GGUFFileReader::new(cursor).expect("Failed to read");
        
        let tensor_info = reader.get_tensor_info(&name).unwrap();
        prop_assert_eq!(tensor_info.tensor_type(), TensorType::I32);
        prop_assert_eq!(tensor_info.shape().dimensions(), shape.as_slice());
        
        let loaded_data = reader.load_tensor_data(&name)
            .expect("Failed to load")
            .expect("Should exist");
        
        let loaded_ints: Vec<i32> = loaded_data.as_slice()
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        prop_assert_eq!(loaded_ints, data);
    }
    
    #[test]
    fn test_multiple_tensors_round_trip(
        tensors in prop::collection::vec(
            (tensor_name_strategy(), tensor_shape_strategy()),
            1..10
        )
    ) {
        // Ensure unique names
        let mut unique_names = std::collections::HashSet::new();
        for (name, _) in &tensors {
            prop_assume!(unique_names.insert(name.clone()));
        }
        
        let mut builder = GGUFBuilder::new();
        let mut expected_data: Vec<(String, Vec<f32>)> = Vec::new();
        
        for (name, shape) in tensors {
            let elements = shape.iter().product::<u64>() as usize;
            prop_assume!(elements <= 1000); // Keep sizes reasonable
            
            let data: Vec<f32> = (0..elements).map(|i| i as f32).collect();
            builder = builder.add_f32_tensor(&name, shape, data.clone());
            expected_data.push((name, data));
        }
        
        let bytes = builder.build_to_bytes().expect("Failed to build");
        
        let cursor = Cursor::new(bytes);
        let mut reader = GGUFFileReader::new(cursor).expect("Failed to read");
        
        prop_assert_eq!(reader.tensor_count() as usize, expected_data.len());
        
        for (name, expected_values) in expected_data {
            let loaded_data = reader.load_tensor_data(&name)
                .expect("Failed to load")
                .expect("Should exist");
                
            let loaded_floats: Vec<f32> = loaded_data.as_slice()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            
            prop_assert_eq!(loaded_floats, expected_values);
        }
    }
    
    #[test]
    fn test_tensor_shape_validation(
        shape in prop::collection::vec(0u64..1000, 1..6)
    ) {
        let element_count = shape.iter().product::<u64>();
        
        // Create data matching the shape
        let data = vec![1.0f32; element_count as usize];
        
        let mut builder = GGUFBuilder::new();
        builder = builder.add_f32_tensor("test", shape.clone(), data);
        
        let bytes = builder.build_to_bytes().expect("Failed to build");
        let cursor = Cursor::new(bytes);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read");
        
        let tensor_info = reader.get_tensor_info("test").unwrap();
        prop_assert_eq!(tensor_info.shape().dimensions(), shape.as_slice());
        prop_assert_eq!(tensor_info.element_count(), element_count);
        prop_assert_eq!(tensor_info.shape().rank() as usize, shape.len());
    }
    
    #[test]
    fn test_empty_tensors(
        shape in prop::collection::vec(0u64..10, 1..4)
            .prop_filter("At least one dimension must be 0", |s| s.iter().any(|&x| x == 0))
    ) {
        let element_count = shape.iter().product::<u64>();
        prop_assert_eq!(element_count, 0);
        
        let data = vec![]; // Empty data
        
        let mut builder = GGUFBuilder::new();
        builder = builder.add_f32_tensor("empty", shape.clone(), data);
        
        let bytes = builder.build_to_bytes().expect("Failed to build");
        let cursor = Cursor::new(bytes);
        let mut reader = GGUFFileReader::new(cursor).expect("Failed to read");
        
        let tensor_info = reader.get_tensor_info("empty").unwrap();
        prop_assert_eq!(tensor_info.element_count(), 0);
        prop_assert_eq!(tensor_info.size_in_bytes(), 0);
        
        let loaded_data = reader.load_tensor_data("empty")
            .expect("Failed to load")
            .expect("Should exist");
        prop_assert_eq!(loaded_data.len(), 0);
    }
}

// Additional tests for edge cases and invariants
proptest! {
    #[test]
    fn test_tensor_data_size_invariants(
        tensor_type in prop_oneof![
            Just(TensorType::F32),
            Just(TensorType::F16), 
            Just(TensorType::I32),
            Just(TensorType::I16),
            Just(TensorType::I8),
        ],
        element_count in 1u64..1000
    ) {
        let shape = vec![element_count];
        let expected_bytes = element_count * tensor_type.size_in_bytes() as u64;
        
        // Create appropriate data
        let data = match tensor_type {
            TensorType::F32 => TensorData::F32(vec![0.0f32; element_count as usize]),
            TensorType::I32 => TensorData::I32(vec![0i32; element_count as usize]),
            _ => TensorData::Bytes(vec![0u8; expected_bytes as usize]),
        };
        
        let tensor_info = TensorInfo::new(
            "test".to_string(),
            tensor_type,
            TensorShape::new(shape),
            0
        );
        
        prop_assert_eq!(tensor_info.size_in_bytes(), expected_bytes);
        prop_assert_eq!(tensor_info.element_count(), element_count);
        prop_assert_eq!(data.len() as u64, expected_bytes);
    }
    
    #[test]
    fn test_quantized_tensor_block_alignment(
        element_count in 32u64..10000,  // Ensure we have at least one block
        quant_type in prop_oneof![
            Just(TensorType::Q4_0),
            Just(TensorType::Q4_1),
            Just(TensorType::Q8_0),
        ]
    ) {
        let block_size = quant_type.block_size() as u64;
        let block_count = (element_count + block_size - 1) / block_size; // Round up
        let expected_data_size = block_count * quant_type.size_in_bytes() as u64;
        
        // Create mock quantized data of the right size
        let quantized_data = vec![0u8; expected_data_size as usize];
        
        let mut builder = GGUFBuilder::new();
        builder = builder.add_quantized_tensor(
            "quantized", 
            vec![element_count], 
            quant_type, 
            quantized_data
        );
        
        let bytes = builder.build_to_bytes().expect("Failed to build");
        let cursor = Cursor::new(bytes);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read");
        
        let tensor_info = reader.get_tensor_info("quantized").unwrap();
        prop_assert_eq!(tensor_info.tensor_type(), quant_type);
        prop_assert_eq!(tensor_info.element_count(), element_count);
        
        // The size should be block-aligned
        let actual_size = tensor_info.size_in_bytes();
        prop_assert_eq!(actual_size, expected_data_size);
        prop_assert_eq!(actual_size % quant_type.size_in_bytes() as u64, 0);
    }
}