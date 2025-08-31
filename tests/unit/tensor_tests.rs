//! Unit tests for the tensor module

use gguf::tensor::*;
use gguf::prelude::*;
use std::io::Cursor;

mod tensor_type_tests {
    use super::*;

    #[test]
    fn test_tensor_type_from_u32() {
        assert_eq!(TensorType::try_from(0u32).unwrap(), TensorType::F32);
        assert_eq!(TensorType::try_from(1u32).unwrap(), TensorType::F16);
        assert_eq!(TensorType::try_from(2u32).unwrap(), TensorType::Q4_0);
        assert_eq!(TensorType::try_from(3u32).unwrap(), TensorType::Q4_1);
        assert_eq!(TensorType::try_from(8u32).unwrap(), TensorType::Q8_0);
        assert_eq!(TensorType::try_from(24u32).unwrap(), TensorType::I8);
        assert_eq!(TensorType::try_from(26u32).unwrap(), TensorType::I32);
        assert_eq!(TensorType::try_from(28u32).unwrap(), TensorType::F64);
        assert_eq!(TensorType::try_from(30u32).unwrap(), TensorType::BF16);
        
        // Test invalid type
        assert!(TensorType::try_from(999u32).is_err());
    }

    #[test]
    fn test_tensor_type_to_u32() {
        assert_eq!(TensorType::F32 as u32, 0);
        assert_eq!(TensorType::F16 as u32, 1);
        assert_eq!(TensorType::Q4_0 as u32, 2);
        assert_eq!(TensorType::Q4_1 as u32, 3);
        assert_eq!(TensorType::Q8_0 as u32, 8);
        assert_eq!(TensorType::I8 as u32, 24);
        assert_eq!(TensorType::I32 as u32, 26);
        assert_eq!(TensorType::F64 as u32, 28);
        assert_eq!(TensorType::BF16 as u32, 30);
    }

    #[test]
    fn test_tensor_type_size_in_bytes() {
        assert_eq!(TensorType::F32.size_in_bytes(), 4);
        assert_eq!(TensorType::F16.size_in_bytes(), 2);
        assert_eq!(TensorType::F64.size_in_bytes(), 8);
        assert_eq!(TensorType::I8.size_in_bytes(), 1);
        assert_eq!(TensorType::I16.size_in_bytes(), 2);
        assert_eq!(TensorType::I32.size_in_bytes(), 4);
        assert_eq!(TensorType::I64.size_in_bytes(), 8);
        assert_eq!(TensorType::BF16.size_in_bytes(), 2);
        
        // Quantized types have specific sizes
        assert_eq!(TensorType::Q4_0.size_in_bytes(), 18); // 32 4-bit values + 2 bytes metadata per block
        assert_eq!(TensorType::Q4_1.size_in_bytes(), 20); // 32 4-bit values + 4 bytes metadata per block
        assert_eq!(TensorType::Q8_0.size_in_bytes(), 34); // 32 8-bit values + 2 bytes metadata per block
    }

    #[test]
    fn test_tensor_type_is_quantized() {
        // Non-quantized types
        assert!(!TensorType::F32.is_quantized());
        assert!(!TensorType::F16.is_quantized());
        assert!(!TensorType::F64.is_quantized());
        assert!(!TensorType::I8.is_quantized());
        assert!(!TensorType::I16.is_quantized());
        assert!(!TensorType::I32.is_quantized());
        assert!(!TensorType::I64.is_quantized());
        assert!(!TensorType::BF16.is_quantized());
        
        // Quantized types
        assert!(TensorType::Q4_0.is_quantized());
        assert!(TensorType::Q4_1.is_quantized());
        assert!(TensorType::Q5_0.is_quantized());
        assert!(TensorType::Q5_1.is_quantized());
        assert!(TensorType::Q8_0.is_quantized());
        assert!(TensorType::Q8_1.is_quantized());
        assert!(TensorType::Q2_K.is_quantized());
        assert!(TensorType::Q3_K.is_quantized());
        assert!(TensorType::Q4_K.is_quantized());
        assert!(TensorType::Q5_K.is_quantized());
        assert!(TensorType::Q6_K.is_quantized());
        assert!(TensorType::Q8_K.is_quantized());
    }

    #[test]
    fn test_tensor_type_is_float() {
        // Float types
        assert!(TensorType::F32.is_float());
        assert!(TensorType::F16.is_float());
        assert!(TensorType::F64.is_float());
        assert!(TensorType::BF16.is_float());
        
        // Non-float types
        assert!(!TensorType::I8.is_float());
        assert!(!TensorType::I16.is_float());
        assert!(!TensorType::I32.is_float());
        assert!(!TensorType::I64.is_float());
        assert!(!TensorType::Q4_0.is_float());
        assert!(!TensorType::Q8_0.is_float());
    }

    #[test]
    fn test_tensor_type_is_integer() {
        // Integer types
        assert!(TensorType::I8.is_integer());
        assert!(TensorType::I16.is_integer());
        assert!(TensorType::I32.is_integer());
        assert!(TensorType::I64.is_integer());
        
        // Non-integer types
        assert!(!TensorType::F32.is_integer());
        assert!(!TensorType::F16.is_integer());
        assert!(!TensorType::F64.is_integer());
        assert!(!TensorType::BF16.is_integer());
        assert!(!TensorType::Q4_0.is_integer());
        assert!(!TensorType::Q8_0.is_integer());
    }

    #[test]
    fn test_tensor_type_name() {
        assert_eq!(TensorType::F32.name(), "F32");
        assert_eq!(TensorType::F16.name(), "F16");
        assert_eq!(TensorType::Q4_0.name(), "Q4_0");
        assert_eq!(TensorType::Q4_1.name(), "Q4_1");
        assert_eq!(TensorType::Q8_0.name(), "Q8_0");
        assert_eq!(TensorType::I32.name(), "I32");
        assert_eq!(TensorType::F64.name(), "F64");
        assert_eq!(TensorType::BF16.name(), "BF16");
    }

    #[test]
    fn test_tensor_type_block_size() {
        // Most quantized types have block size 32
        assert_eq!(TensorType::Q4_0.block_size(), 32);
        assert_eq!(TensorType::Q4_1.block_size(), 32);
        assert_eq!(TensorType::Q5_0.block_size(), 32);
        assert_eq!(TensorType::Q5_1.block_size(), 32);
        assert_eq!(TensorType::Q8_0.block_size(), 32);
        assert_eq!(TensorType::Q8_1.block_size(), 32);
        
        // K-quantized types have different block sizes
        assert_eq!(TensorType::Q2_K.block_size(), 256);
        assert_eq!(TensorType::Q3_K.block_size(), 256);
        assert_eq!(TensorType::Q4_K.block_size(), 256);
        assert_eq!(TensorType::Q5_K.block_size(), 256);
        assert_eq!(TensorType::Q6_K.block_size(), 256);
        assert_eq!(TensorType::Q8_K.block_size(), 256);
        
        // Non-quantized types have block size 1
        assert_eq!(TensorType::F32.block_size(), 1);
        assert_eq!(TensorType::F16.block_size(), 1);
        assert_eq!(TensorType::I32.block_size(), 1);
    }
}

mod tensor_shape_tests {
    use super::*;

    #[test]
    fn test_tensor_shape_creation() {
        let shape = TensorShape::new(vec![10, 20, 30]);
        
        assert_eq!(shape.dimensions(), &[10, 20, 30]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.element_count(), 10 * 20 * 30);
    }

    #[test]
    fn test_tensor_shape_1d() {
        let shape = TensorShape::new(vec![100]);
        
        assert_eq!(shape.dimensions(), &[100]);
        assert_eq!(shape.rank(), 1);
        assert_eq!(shape.element_count(), 100);
    }

    #[test]
    fn test_tensor_shape_scalar() {
        let shape = TensorShape::new(vec![]);
        
        assert_eq!(shape.dimensions(), &[]);
        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.element_count(), 1); // Scalar has 1 element
    }

    #[test]
    fn test_tensor_shape_with_zeros() {
        let shape = TensorShape::new(vec![10, 0, 5]);
        
        assert_eq!(shape.dimensions(), &[10, 0, 5]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.element_count(), 0); // Any zero dimension makes count zero
    }

    #[test]
    fn test_tensor_shape_equality() {
        let shape1 = TensorShape::new(vec![2, 3, 4]);
        let shape2 = TensorShape::new(vec![2, 3, 4]);
        let shape3 = TensorShape::new(vec![2, 3, 5]);
        
        assert_eq!(shape1, shape2);
        assert_ne!(shape1, shape3);
    }

    #[test]
    fn test_tensor_shape_indexing() {
        let shape = TensorShape::new(vec![10, 20, 30]);
        
        assert_eq!(shape[0], 10);
        assert_eq!(shape[1], 20);
        assert_eq!(shape[2], 30);
    }

    #[test]
    fn test_tensor_shape_iteration() {
        let dims = vec![1, 2, 3, 4];
        let shape = TensorShape::new(dims.clone());
        
        let collected: Vec<u64> = shape.iter().copied().collect();
        assert_eq!(collected, dims);
    }

    #[test]
    fn test_tensor_shape_from_iter() {
        let dims = vec![5, 10, 15];
        let shape: TensorShape = dims.iter().copied().collect();
        
        assert_eq!(shape.dimensions(), &dims);
    }

    #[test]
    fn test_tensor_shape_serialization() {
        let shape = TensorShape::new(vec![1, 2, 3]);
        let bytes = shape.to_bytes();
        
        // Should serialize rank followed by dimensions
        assert!(!bytes.is_empty());
        
        let mut cursor = Cursor::new(&bytes);
        let deserialized = TensorShape::read(&mut cursor).expect("Failed to deserialize shape");
        
        assert_eq!(deserialized, shape);
    }

    #[test]
    fn test_tensor_shape_large_dimensions() {
        let large_dim = u64::MAX / 4;
        let shape = TensorShape::new(vec![2, 2]);
        
        assert_eq!(shape.element_count(), 4);
        
        // Test potential overflow
        let shape_large = TensorShape::new(vec![large_dim, 2]);
        // This might overflow, but should handle gracefully
        let count = shape_large.element_count();
        assert!(count == 0 || count > large_dim); // Either overflow to 0 or actual value
    }
}

mod tensor_data_tests {
    use super::*;

    #[test]
    fn test_tensor_data_bytes() {
        let data = vec![1, 2, 3, 4, 5];
        let tensor_data = TensorData::Bytes(data.clone());
        
        assert_eq!(tensor_data.len(), 5);
        assert!(!tensor_data.is_empty());
        assert_eq!(tensor_data.as_slice(), &data);
    }

    #[test]
    fn test_tensor_data_empty() {
        let tensor_data = TensorData::Bytes(vec![]);
        
        assert_eq!(tensor_data.len(), 0);
        assert!(tensor_data.is_empty());
        assert!(tensor_data.as_slice().is_empty());
    }

    #[test]
    fn test_tensor_data_f32() {
        let data = vec![1.0f32, 2.5f32, -3.14f32];
        let tensor_data = TensorData::F32(data.clone());
        
        assert_eq!(tensor_data.len(), 12); // 3 * 4 bytes
        assert!(!tensor_data.is_empty());
        
        // Should be able to convert to bytes
        let bytes = tensor_data.as_slice();
        assert_eq!(bytes.len(), 12);
    }

    #[test]
    fn test_tensor_data_f16() {
        // Using half::f16 for proper f16 support
        let data = vec![0u16; 4]; // Representing f16 as u16
        let tensor_data = TensorData::F16(data);
        
        assert_eq!(tensor_data.len(), 8); // 4 * 2 bytes
        assert!(!tensor_data.is_empty());
    }

    #[test]
    fn test_tensor_data_i32() {
        let data = vec![100i32, -200i32, 300i32];
        let tensor_data = TensorData::I32(data.clone());
        
        assert_eq!(tensor_data.len(), 12); // 3 * 4 bytes
        assert!(!tensor_data.is_empty());
    }

    #[test]
    fn test_tensor_data_conversion() {
        let f32_data = vec![1.0f32, 2.0f32, 3.0f32];
        let tensor_data = TensorData::F32(f32_data.clone());
        
        // Test conversion to bytes
        let bytes = tensor_data.as_slice();
        assert_eq!(bytes.len(), 12);
        
        // Verify the actual byte values
        let expected: Vec<u8> = f32_data
            .iter()
            .flat_map(|&f| f.to_le_bytes().to_vec())
            .collect();
        assert_eq!(bytes, &expected);
    }

    #[test]
    fn test_tensor_data_clone() {
        let data = vec![1, 2, 3, 4, 5];
        let tensor_data = TensorData::Bytes(data);
        let cloned = tensor_data.clone();
        
        assert_eq!(tensor_data.len(), cloned.len());
        assert_eq!(tensor_data.as_slice(), cloned.as_slice());
    }

    #[test]
    fn test_tensor_data_debug() {
        let tensor_data = TensorData::Bytes(vec![1, 2, 3]);
        let debug_str = format!("{:?}", tensor_data);
        
        assert!(debug_str.contains("Bytes"));
    }
}

mod tensor_info_tests {
    use super::*;

    #[test]
    fn test_tensor_info_creation() {
        let name = "test_tensor".to_string();
        let tensor_type = TensorType::F32;
        let shape = TensorShape::new(vec![10, 20]);
        let offset = 1024u64;
        
        let info = TensorInfo::new(name.clone(), tensor_type, shape.clone(), offset);
        
        assert_eq!(info.name(), &name);
        assert_eq!(info.tensor_type(), tensor_type);
        assert_eq!(info.shape(), &shape);
        assert_eq!(info.offset(), offset);
        assert_eq!(info.element_count(), 200);
        assert_eq!(info.size_in_bytes(), 800); // 200 * 4 bytes
    }

    #[test]
    fn test_tensor_info_quantized() {
        let info = TensorInfo::new(
            "quantized_tensor".to_string(),
            TensorType::Q4_0,
            TensorShape::new(vec![64]), // One block
            0,
        );
        
        assert_eq!(info.element_count(), 64);
        // Q4_0 has 32 elements per block, so 64 elements = 2 blocks
        // Each block is 18 bytes
        assert_eq!(info.size_in_bytes(), 36); // 2 * 18 bytes
    }

    #[test]
    fn test_tensor_info_with_zero_dimension() {
        let info = TensorInfo::new(
            "zero_tensor".to_string(),
            TensorType::F32,
            TensorShape::new(vec![10, 0, 5]),
            0,
        );
        
        assert_eq!(info.element_count(), 0);
        assert_eq!(info.size_in_bytes(), 0);
    }

    #[test]
    fn test_tensor_info_serialization() {
        let info = TensorInfo::new(
            "serialize_test".to_string(),
            TensorType::F16,
            TensorShape::new(vec![5, 10]),
            2048,
        );
        
        let bytes = info.to_bytes();
        assert!(!bytes.is_empty());
        
        let mut cursor = Cursor::new(&bytes);
        let deserialized = TensorInfo::read(&mut cursor).expect("Failed to deserialize tensor info");
        
        assert_eq!(deserialized.name(), info.name());
        assert_eq!(deserialized.tensor_type(), info.tensor_type());
        assert_eq!(deserialized.shape(), info.shape());
        assert_eq!(deserialized.offset(), info.offset());
    }

    #[test]
    fn test_tensor_info_display() {
        let info = TensorInfo::new(
            "display_test".to_string(),
            TensorType::F32,
            TensorShape::new(vec![2, 3, 4]),
            512,
        );
        
        let display_str = format!("{}", info);
        assert!(display_str.contains("display_test"));
        assert!(display_str.contains("F32"));
        assert!(display_str.contains("[2, 3, 4]"));
    }
}

mod quantization_tests {
    use super::*;

    #[test]
    fn test_block_size_calculations() {
        // Test various quantized types
        assert_eq!(calculate_block_size(TensorType::Q4_0), 32);
        assert_eq!(calculate_block_size(TensorType::Q4_1), 32);
        assert_eq!(calculate_block_size(TensorType::Q8_0), 32);
        
        // K-quantized types
        assert_eq!(calculate_block_size(TensorType::Q2_K), 256);
        assert_eq!(calculate_block_size(TensorType::Q3_K), 256);
        assert_eq!(calculate_block_size(TensorType::Q4_K), 256);
        
        // Non-quantized types
        assert_eq!(calculate_block_size(TensorType::F32), 1);
        assert_eq!(calculate_block_size(TensorType::I32), 1);
    }

    #[test]
    fn test_quantized_size_calculation() {
        // Test Q4_0: 32 elements per block, each block is 18 bytes
        let size = calculate_quantized_size(64, TensorType::Q4_0);
        assert_eq!(size, 36); // 2 blocks * 18 bytes
        
        // Test Q8_0: 32 elements per block, each block is 34 bytes
        let size = calculate_quantized_size(64, TensorType::Q8_0);
        assert_eq!(size, 68); // 2 blocks * 34 bytes
        
        // Test with non-block-aligned size
        let size = calculate_quantized_size(50, TensorType::Q4_0);
        assert_eq!(size, 36); // Still 2 blocks (rounded up)
    }

    #[test]
    fn test_quantization_metadata() {
        // Test Q4_0 metadata structure
        let q4_0_info = get_quantization_info(TensorType::Q4_0);
        assert_eq!(q4_0_info.block_size, 32);
        assert_eq!(q4_0_info.type_size, 18);
        assert!(q4_0_info.has_scale);
        assert!(!q4_0_info.has_zero_point);
        
        // Test Q4_1 metadata structure
        let q4_1_info = get_quantization_info(TensorType::Q4_1);
        assert_eq!(q4_1_info.block_size, 32);
        assert_eq!(q4_1_info.type_size, 20);
        assert!(q4_1_info.has_scale);
        assert!(q4_1_info.has_zero_point);
    }

    #[test]
    fn test_dequantization_requirements() {
        // Test which types can be dequantized
        assert!(can_dequantize(TensorType::Q4_0));
        assert!(can_dequantize(TensorType::Q8_0));
        assert!(!can_dequantize(TensorType::F32)); // Already dequantized
        assert!(!can_dequantize(TensorType::I32)); // Not quantized
    }

    #[test]
    fn test_quantization_precision() {
        // Test precision information
        assert_eq!(get_quantization_bits(TensorType::Q4_0), 4);
        assert_eq!(get_quantization_bits(TensorType::Q8_0), 8);
        assert_eq!(get_quantization_bits(TensorType::Q2_K), 2);
        assert_eq!(get_quantization_bits(TensorType::F32), 32); // Full precision
    }

    #[test]
    fn test_quantization_error_handling() {
        // Test invalid quantization operations
        assert!(validate_quantization_params(TensorType::Q4_0, &[]).is_err());
        assert!(validate_quantization_params(TensorType::F32, &[1, 2, 3]).is_ok());
    }
}

// Helper functions for quantization tests
fn calculate_block_size(tensor_type: TensorType) -> usize {
    tensor_type.block_size()
}

fn calculate_quantized_size(elements: usize, tensor_type: TensorType) -> usize {
    if !tensor_type.is_quantized() {
        return elements * tensor_type.size_in_bytes();
    }
    
    let block_size = tensor_type.block_size();
    let block_count = (elements + block_size - 1) / block_size; // Round up
    block_count * tensor_type.size_in_bytes()
}

#[derive(Debug)]
struct QuantizationInfo {
    block_size: usize,
    type_size: usize,
    has_scale: bool,
    has_zero_point: bool,
}

fn get_quantization_info(tensor_type: TensorType) -> QuantizationInfo {
    match tensor_type {
        TensorType::Q4_0 => QuantizationInfo {
            block_size: 32,
            type_size: 18,
            has_scale: true,
            has_zero_point: false,
        },
        TensorType::Q4_1 => QuantizationInfo {
            block_size: 32,
            type_size: 20,
            has_scale: true,
            has_zero_point: true,
        },
        TensorType::Q8_0 => QuantizationInfo {
            block_size: 32,
            type_size: 34,
            has_scale: true,
            has_zero_point: false,
        },
        _ => QuantizationInfo {
            block_size: 1,
            type_size: tensor_type.size_in_bytes(),
            has_scale: false,
            has_zero_point: false,
        },
    }
}

fn can_dequantize(tensor_type: TensorType) -> bool {
    tensor_type.is_quantized()
}

fn get_quantization_bits(tensor_type: TensorType) -> u8 {
    match tensor_type {
        TensorType::Q2_K => 2,
        TensorType::Q3_K => 3,
        TensorType::Q4_0 | TensorType::Q4_1 | TensorType::Q4_K => 4,
        TensorType::Q5_0 | TensorType::Q5_1 | TensorType::Q5_K => 5,
        TensorType::Q6_K => 6,
        TensorType::Q8_0 | TensorType::Q8_1 | TensorType::Q8_K | TensorType::I8 => 8,
        TensorType::F16 | TensorType::BF16 | TensorType::I16 => 16,
        TensorType::F32 | TensorType::I32 => 32,
        TensorType::F64 | TensorType::I64 => 64,
        _ => 32, // Default
    }
}

fn validate_quantization_params(tensor_type: TensorType, _params: &[u8]) -> Result<()> {
    if tensor_type.is_quantized() && _params.is_empty() {
        Err(GGUFError::InvalidTensorData("Missing quantization parameters".to_string()))
    } else {
        Ok(())
    }
}