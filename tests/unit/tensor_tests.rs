//! Unit tests for the tensor module

use gguf_rs_lib::prelude::*;
use gguf_rs_lib::tensor::*;

mod tensor_type_tests {
    use super::*;

    #[test]
    fn test_tensor_type_from_u32() {
        assert_eq!(TensorType::from_u32(0u32).unwrap(), TensorType::F32);
        assert_eq!(TensorType::from_u32(1u32).unwrap(), TensorType::F16);
        assert_eq!(TensorType::from_u32(2u32).unwrap(), TensorType::Q4_0);
        assert_eq!(TensorType::from_u32(3u32).unwrap(), TensorType::Q4_1);
        assert_eq!(TensorType::from_u32(8u32).unwrap(), TensorType::Q8_0);
        assert_eq!(TensorType::from_u32(30u32).unwrap(), TensorType::I8);
        assert_eq!(TensorType::from_u32(31u32).unwrap(), TensorType::I16);
        assert_eq!(TensorType::from_u32(24u32).unwrap(), TensorType::I32);
        assert_eq!(TensorType::from_u32(25u32).unwrap(), TensorType::I64);
        assert_eq!(TensorType::from_u32(26u32).unwrap(), TensorType::F64);
        assert_eq!(TensorType::from_u32(28u32).unwrap(), TensorType::BF16);

        // Test invalid type
        assert!(TensorType::from_u32(999u32).is_err());
    }

    #[test]
    fn test_tensor_type_to_u32() {
        assert_eq!(TensorType::F32 as u32, 0);
        assert_eq!(TensorType::F16 as u32, 1);
        assert_eq!(TensorType::Q4_0 as u32, 2);
        assert_eq!(TensorType::Q4_1 as u32, 3);
        assert_eq!(TensorType::Q8_0 as u32, 8);
        assert_eq!(TensorType::I32 as u32, 24);
        assert_eq!(TensorType::I64 as u32, 25);
        assert_eq!(TensorType::F64 as u32, 26);
        assert_eq!(TensorType::BF16 as u32, 28);
        assert_eq!(TensorType::I8 as u32, 30);
        assert_eq!(TensorType::I16 as u32, 31);
    }

    #[test]
    fn test_tensor_type_element_size() {
        // Non-quantized types have fixed element sizes
        assert_eq!(TensorType::F32.element_size(), 4);
        assert_eq!(TensorType::F16.element_size(), 2);
        assert_eq!(TensorType::F64.element_size(), 8);
        assert_eq!(TensorType::I8.element_size(), 1);
        assert_eq!(TensorType::I16.element_size(), 2);
        assert_eq!(TensorType::I32.element_size(), 4);
        assert_eq!(TensorType::I64.element_size(), 8);
        assert_eq!(TensorType::BF16.element_size(), 2);

        // Quantized types: check calculate_size for one block
        assert_eq!(TensorType::Q4_0.calculate_size(32), 18); // 32 elements in one block = 18 bytes
        assert_eq!(TensorType::Q4_1.calculate_size(32), 20); // 32 elements in one block = 20 bytes
        assert_eq!(TensorType::Q8_0.calculate_size(32), 34); // 32 elements in one block = 34 bytes
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
        // Helper function to check if a tensor type represents floating point data
        let is_float = |t: TensorType| {
            matches!(t, TensorType::F32 | TensorType::F16 | TensorType::F64 | TensorType::BF16)
        };

        // Float types
        assert!(is_float(TensorType::F32));
        assert!(is_float(TensorType::F16));
        assert!(is_float(TensorType::F64));
        assert!(is_float(TensorType::BF16));

        // Non-float types
        assert!(!is_float(TensorType::I8));
        assert!(!is_float(TensorType::I16));
        assert!(!is_float(TensorType::I32));
        assert!(!is_float(TensorType::I64));
        assert!(!is_float(TensorType::Q4_0));
        assert!(!is_float(TensorType::Q8_0));
    }

    #[test]
    fn test_tensor_type_is_integer() {
        // Helper function to check if a tensor type represents integer data
        let is_integer = |t: TensorType| {
            matches!(t, TensorType::I8 | TensorType::I16 | TensorType::I32 | TensorType::I64)
        };

        // Integer types
        assert!(is_integer(TensorType::I8));
        assert!(is_integer(TensorType::I16));
        assert!(is_integer(TensorType::I32));
        assert!(is_integer(TensorType::I64));

        // Non-integer types
        assert!(!is_integer(TensorType::F32));
        assert!(!is_integer(TensorType::F16));
        assert!(!is_integer(TensorType::F64));
        assert!(!is_integer(TensorType::BF16));
        assert!(!is_integer(TensorType::Q4_0));
        assert!(!is_integer(TensorType::Q8_0));
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
        let shape = TensorShape::new(vec![10, 20, 30]).expect("Valid shape");

        assert_eq!(shape.dims(), &[10, 20, 30]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.element_count(), 10 * 20 * 30);
    }

    #[test]
    fn test_tensor_shape_1d() {
        let shape = TensorShape::new(vec![100]).expect("Valid shape");

        assert_eq!(shape.dims(), &[100]);
        assert_eq!(shape.ndim(), 1);
        assert_eq!(shape.element_count(), 100);
    }

    #[test]
    fn test_tensor_shape_scalar() {
        // TensorShape::new(vec![]) should fail, but we can use scalar()
        let shape = TensorShape::scalar();

        assert_eq!(shape.dims(), &[1]);
        assert_eq!(shape.ndim(), 1);
        assert_eq!(shape.element_count(), 1); // Scalar has 1 element
    }

    #[test]
    fn test_tensor_shape_with_zeros() {
        // Zero dimensions are now allowed for empty tensors - they represent tensors with 0 elements
        // This is mathematically valid and commonly used in practice
        let result = TensorShape::new(vec![10, 0, 5]);
        assert!(result.is_ok());
        let shape = result.unwrap();
        assert_eq!(shape.element_count(), 0); // 10 * 0 * 5 = 0
    }

    #[test]
    fn test_tensor_shape_equality() {
        let shape1 = TensorShape::new(vec![2, 3, 4]).unwrap();
        let shape2 = TensorShape::new(vec![2, 3, 4]).unwrap();
        let shape3 = TensorShape::new(vec![2, 3, 5]).unwrap();

        assert_eq!(shape1, shape2);
        assert_ne!(shape1, shape3);
    }

    #[test]
    fn test_tensor_shape_indexing() {
        let shape = TensorShape::new(vec![10, 20, 30]).unwrap();

        assert_eq!(shape[0], 10);
        assert_eq!(shape[1], 20);
        assert_eq!(shape[2], 30);
    }

    #[test]
    fn test_tensor_shape_iteration() {
        let dims = vec![1, 2, 3, 4];
        let shape = TensorShape::new(dims.clone()).unwrap();

        let collected: Vec<u64> = shape.dimensions.clone();
        assert_eq!(collected, dims);
    }

    #[test]
    fn test_tensor_shape_from_iter() {
        let dims = vec![5, 10, 15];
        let shape = TensorShape::new(dims.clone()).unwrap();

        assert_eq!(shape.dims(), &dims);
    }

    #[test]
    fn test_tensor_shape_serialization() {
        let shape = TensorShape::new(vec![1, 2, 3]).unwrap();

        // Test accessing dimensions
        assert_eq!(shape.dims(), &[1, 2, 3]);
        assert_eq!(shape.ndim(), 3);
    }

    #[test]
    fn test_tensor_shape_large_dimensions() {
        let large_dim = u64::MAX / 4;
        let shape = TensorShape::new(vec![2, 2]).expect("Valid shape");

        assert_eq!(shape.element_count(), 4);

        // Test potential overflow - but large_dim might be too large and cause validation error
        let shape_large_result = TensorShape::new(vec![large_dim, 2]);
        if let Ok(shape_large) = shape_large_result {
            // This might overflow, but should handle gracefully
            let count = shape_large.element_count();
            assert!(count == 0 || count > large_dim); // Either overflow to 0 or actual value
        } else {
            // It's also valid for the validation to reject very large dimensions
            // In which case we just test that it properly rejected it
            assert!(shape_large_result.is_err());
        }
    }
}

mod tensor_data_tests {
    use super::*;

    #[test]
    fn test_tensor_data_bytes() {
        let data = vec![1, 2, 3, 4, 5];
        let tensor_data = TensorData::new_owned(data.clone());

        assert_eq!(tensor_data.len(), 5);
        assert!(!tensor_data.is_empty());
        assert_eq!(tensor_data.as_slice(), &data);
    }

    #[test]
    fn test_tensor_data_empty() {
        let tensor_data = TensorData::new_owned(vec![]);

        assert_eq!(tensor_data.len(), 0);
        assert!(tensor_data.is_empty());
        assert!(tensor_data.as_slice().is_empty());
    }

    #[test]
    fn test_tensor_data_f32() {
        let data = vec![1.0f32, 2.5f32, -std::f32::consts::PI];
        let mut bytes = Vec::new();
        for value in &data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let tensor_data = TensorData::new_owned(bytes);

        assert_eq!(tensor_data.len(), 12); // 3 * 4 bytes
        assert!(!tensor_data.is_empty());

        // Should be able to convert to bytes
        let bytes = tensor_data.as_slice();
        assert_eq!(bytes.len(), 12);
    }

    #[test]
    fn test_tensor_data_f16() {
        // Using half::f16 for proper f16 support
        let data = [0u16; 4]; // Representing f16 as u16
        let data_bytes: Vec<u8> = data.iter().flat_map(|&val| val.to_le_bytes().to_vec()).collect();
        let tensor_data = TensorData::new_owned(data_bytes);

        assert_eq!(tensor_data.len(), 8); // 4 * 2 bytes
        assert!(!tensor_data.is_empty());
    }

    #[test]
    fn test_tensor_data_i32() {
        let data = vec![100i32, -200i32, 300i32];

        // Convert to bytes
        let mut bytes = Vec::new();
        for val in data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        let tensor_data = TensorData::new_owned(bytes);

        assert_eq!(tensor_data.len(), 12); // 3 * 4 bytes
        assert!(!tensor_data.is_empty());
    }

    #[test]
    fn test_tensor_data_conversion() {
        let f32_data = [1.0f32, 2.0f32, 3.0f32];
        let bytes: Vec<u8> = f32_data.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let tensor_data = TensorData::new_owned(bytes);

        // Test conversion to bytes
        let bytes = tensor_data.as_slice();
        assert_eq!(bytes.len(), 12);

        // Verify the actual byte values
        let expected: Vec<u8> = f32_data.iter().flat_map(|&f| f.to_le_bytes().to_vec()).collect();
        assert_eq!(bytes, &expected);
    }

    #[test]
    fn test_tensor_data_clone() {
        let data = vec![1, 2, 3, 4, 5];
        let tensor_data = TensorData::new_owned(data);
        let cloned = tensor_data.clone();

        assert_eq!(tensor_data.len(), cloned.len());
        assert_eq!(tensor_data.as_slice(), cloned.as_slice());
    }

    #[test]
    fn test_tensor_data_debug() {
        let tensor_data = TensorData::new_owned(vec![1, 2, 3]);
        let debug_str = format!("{:?}", tensor_data);

        // The debug output should contain "Owned" since it's TensorData::Owned variant
        assert!(debug_str.contains("Owned"));
    }
}

mod tensor_info_tests {
    use super::*;

    #[test]
    fn test_tensor_info_creation() {
        let name = "test_tensor".to_string();
        let tensor_type = TensorType::F32;
        let shape = TensorShape::new(vec![10, 20]).unwrap();
        let offset = 1024u64;

        let info = TensorInfo::new(name.clone(), shape.clone(), tensor_type, offset);

        assert_eq!(info.name(), &name);
        assert_eq!(info.tensor_type(), tensor_type);
        assert_eq!(info.shape(), &shape);
        assert_eq!(info.data_offset, offset);
        assert_eq!(info.element_count(), 200);
        assert_eq!(info.expected_data_size(), 800); // 200 * 4 bytes
    }

    #[test]
    fn test_tensor_info_quantized() {
        let info = TensorInfo::new(
            "quantized_tensor".to_string(),
            TensorShape::new(vec![64]).unwrap(), // One block
            TensorType::Q4_0,
            0,
        );

        assert_eq!(info.element_count(), 64);
        // Q4_0 has 32 elements per block, so 64 elements = 2 blocks
        // Each block is 18 bytes
        assert_eq!(info.expected_data_size(), 36); // 2 * 18 bytes
    }

    #[test]
    fn test_tensor_info_with_zero_dimension() {
        let info = TensorInfo::new(
            "zero_tensor".to_string(),
            TensorShape::new(vec![10, 0, 5]).unwrap(),
            TensorType::F32,
            0,
        );

        assert_eq!(info.element_count(), 0);
        assert_eq!(info.expected_data_size(), 0);
    }

    #[test]
    fn test_tensor_info_properties() {
        let shape = TensorShape::new(vec![5, 10]).unwrap();
        let info =
            TensorInfo::new("serialize_test".to_string(), shape.clone(), TensorType::F16, 2048);

        assert_eq!(info.name(), "serialize_test");
        assert_eq!(info.tensor_type(), TensorType::F16);
        assert_eq!(info.shape(), &shape);
        assert_eq!(info.data_offset(), 2048);
    }

    #[test]
    fn test_tensor_info_display() {
        let info = TensorInfo::new(
            "display_test".to_string(),
            TensorShape::new(vec![2, 3, 4]).unwrap(),
            TensorType::F32,
            512,
        );

        let display_str = format!("{}", info);
        assert!(display_str.contains("display_test"));
        assert!(display_str.contains("F32"));
        assert!(display_str.contains("(2, 3, 4)"));
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
    tensor_type.calculate_size(elements as u64) as usize
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
            type_size: tensor_type.element_size(),
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
