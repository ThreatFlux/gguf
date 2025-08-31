//! Quantization format structures and utilities

use crate::format::types::GGUFTensorType as TensorType;
use serde::{Deserialize, Serialize};

/// Block-based quantization format structures
pub mod blocks {
    /// Q4_0 quantization block (32 4-bit values + scale)
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct Q4_0Block {
        /// FP16 scale factor
        pub scale: [u8; 2], // f16 as bytes
        /// 16 bytes of 4-bit quantized values (32 values, 4 bits each)
        pub data: [u8; 16],
    }

    /// Q4_1 quantization block (32 4-bit values + scale + min)
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct Q4_1Block {
        /// FP16 scale factor
        pub scale: [u8; 2], // f16 as bytes
        /// FP16 minimum value
        pub min: [u8; 2], // f16 as bytes
        /// 16 bytes of 4-bit quantized values
        pub data: [u8; 16],
    }

    /// Q5_0 quantization block (32 5-bit values + scale)
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct Q5_0Block {
        /// FP16 scale factor
        pub scale: [u8; 2],
        /// 4 bytes for high bits (1 bit per value)
        pub high_bits: [u8; 4],
        /// 16 bytes of 4-bit quantized values
        pub data: [u8; 16],
    }

    /// Q5_1 quantization block (32 5-bit values + scale + min)
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct Q5_1Block {
        /// FP16 scale factor
        pub scale: [u8; 2],
        /// FP16 minimum value
        pub min: [u8; 2],
        /// 4 bytes for high bits
        pub high_bits: [u8; 4],
        /// 16 bytes of 4-bit quantized values
        pub data: [u8; 16],
    }

    /// Q8_0 quantization block (32 8-bit values + scale)
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct Q8_0Block {
        /// FP16 scale factor
        pub scale: [u8; 2],
        /// 32 bytes of 8-bit quantized values
        pub data: [u8; 32],
    }

    /// Q8_1 quantization block (32 8-bit values + scale + sum)
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct Q8_1Block {
        /// FP32 scale factor
        pub scale: [u8; 4], // f32 as bytes
        /// 32 bytes of 8-bit quantized values
        pub data: [u8; 32],
    }

    // K-quant blocks are more complex and vary by type
    // These are simplified representations

    /// Q2_K quantization block (256 2-bit values)
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct Q2_KBlock {
        /// Scales and other metadata
        pub metadata: [u8; 18],
        /// Quantized data
        pub data: [u8; 64],
    }

    /// Q3_K quantization block (256 3-bit values)
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct Q3_KBlock {
        /// High bits for extra precision
        pub high_bits: [u8; 32],
        /// Scales and metadata
        pub scales: [u8; 12],
        /// Quantized data
        pub data: [u8; 64],
    }

    /// Q4_K quantization block (256 4-bit values)
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct Q4_KBlock {
        /// Scales
        pub scales: [u8; 12],
        /// Fine scales
        pub fine_scales: [u8; 16],
        /// Quantized data
        pub data: [u8; 128],
    }

    /// Q5_K quantization block (256 5-bit values)
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct Q5_KBlock {
        /// Scales
        pub scales: [u8; 12],
        /// High bits
        pub high_bits: [u8; 32],
        /// Fine scales
        pub fine_scales: [u8; 16],
        /// Quantized data
        pub data: [u8; 128],
    }

    /// Q6_K quantization block (256 6-bit values)
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct Q6_KBlock {
        /// Quantized data (6 bits per value)
        pub data: [u8; 192],
        /// Scales
        pub scales: [u8; 16],
    }

    /// Q8_K quantization block (256 8-bit values)
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct Q8_KBlock {
        /// Quantized data
        pub data: [u8; 256],
        /// Scales
        pub scales: [u8; 32],
    }

    // IQ types are very specialized and implementation-specific
    // These are placeholder structures

    /// IQ2_XXS quantization block (ultra-compressed)
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct IQ2_XXSBlock {
        /// Compressed data (implementation specific)
        pub data: [u8; 8],
    }

    /// IQ3_XXS quantization block
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct IQ3_XXSBlock {
        /// Compressed data (implementation specific)
        pub data: [u8; 12],
    }

    /// IQ4_NL quantization block
    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy)]
    pub struct IQ4_NLBlock {
        /// Compressed data (implementation specific)
        pub data: [u8; 16],
    }
}

/// Quantization parameters for different tensor types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// The tensor type this applies to
    pub tensor_type: TensorType,
    /// Block size (number of elements per quantization block)
    pub block_size: usize,
    /// Bits per weight
    pub bits_per_weight: f32,
    /// Whether this format supports scales
    pub has_scales: bool,
    /// Whether this format supports minimum values
    pub has_min: bool,
    /// Whether this format has high-bit extensions
    pub has_high_bits: bool,
    /// Size of each block in bytes
    pub block_size_bytes: usize,
}

impl QuantizationParams {
    /// Get quantization parameters for a tensor type
    pub fn for_type(tensor_type: TensorType) -> Self {
        use blocks::*;

        match tensor_type {
            TensorType::Q4_0 => Self {
                tensor_type,
                block_size: 32,
                bits_per_weight: 4.0,
                has_scales: true,
                has_min: false,
                has_high_bits: false,
                block_size_bytes: std::mem::size_of::<Q4_0Block>(),
            },
            TensorType::Q4_1 => Self {
                tensor_type,
                block_size: 32,
                bits_per_weight: 4.0,
                has_scales: true,
                has_min: true,
                has_high_bits: false,
                block_size_bytes: std::mem::size_of::<Q4_1Block>(),
            },
            TensorType::Q5_0 => Self {
                tensor_type,
                block_size: 32,
                bits_per_weight: 5.0,
                has_scales: true,
                has_min: false,
                has_high_bits: true,
                block_size_bytes: std::mem::size_of::<Q5_0Block>(),
            },
            TensorType::Q5_1 => Self {
                tensor_type,
                block_size: 32,
                bits_per_weight: 5.0,
                has_scales: true,
                has_min: true,
                has_high_bits: true,
                block_size_bytes: std::mem::size_of::<Q5_1Block>(),
            },
            TensorType::Q8_0 => Self {
                tensor_type,
                block_size: 32,
                bits_per_weight: 8.0,
                has_scales: true,
                has_min: false,
                has_high_bits: false,
                block_size_bytes: std::mem::size_of::<Q8_0Block>(),
            },
            TensorType::Q8_1 => Self {
                tensor_type,
                block_size: 32,
                bits_per_weight: 8.0,
                has_scales: true,
                has_min: false,
                has_high_bits: false,
                block_size_bytes: std::mem::size_of::<Q8_1Block>(),
            },
            TensorType::Q2_K => Self {
                tensor_type,
                block_size: 256,
                bits_per_weight: 2.0,
                has_scales: true,
                has_min: false,
                has_high_bits: false,
                block_size_bytes: std::mem::size_of::<Q2_KBlock>(),
            },
            TensorType::Q3_K => Self {
                tensor_type,
                block_size: 256,
                bits_per_weight: 3.0,
                has_scales: true,
                has_min: false,
                has_high_bits: true,
                block_size_bytes: std::mem::size_of::<Q3_KBlock>(),
            },
            TensorType::Q4_K => Self {
                tensor_type,
                block_size: 256,
                bits_per_weight: 4.0,
                has_scales: true,
                has_min: false,
                has_high_bits: false,
                block_size_bytes: std::mem::size_of::<Q4_KBlock>(),
            },
            TensorType::Q5_K => Self {
                tensor_type,
                block_size: 256,
                bits_per_weight: 5.0,
                has_scales: true,
                has_min: false,
                has_high_bits: true,
                block_size_bytes: std::mem::size_of::<Q5_KBlock>(),
            },
            TensorType::Q6_K => Self {
                tensor_type,
                block_size: 256,
                bits_per_weight: 6.0,
                has_scales: true,
                has_min: false,
                has_high_bits: false,
                block_size_bytes: std::mem::size_of::<Q6_KBlock>(),
            },
            TensorType::Q8_K => Self {
                tensor_type,
                block_size: 256,
                bits_per_weight: 8.0,
                has_scales: true,
                has_min: false,
                has_high_bits: false,
                block_size_bytes: std::mem::size_of::<Q8_KBlock>(),
            },
            // IQ types - these are approximate/placeholder values
            TensorType::IQ1_S | TensorType::IQ1_M => Self {
                tensor_type,
                block_size: 32,
                bits_per_weight: 1.0,
                has_scales: true,
                has_min: false,
                has_high_bits: false,
                block_size_bytes: 8, // Approximate
            },
            TensorType::IQ2_XXS | TensorType::IQ2_XS | TensorType::IQ2_S => Self {
                tensor_type,
                block_size: 32,
                bits_per_weight: 2.0,
                has_scales: true,
                has_min: false,
                has_high_bits: false,
                block_size_bytes: std::mem::size_of::<IQ2_XXSBlock>(),
            },
            TensorType::IQ3_XXS | TensorType::IQ3_S => Self {
                tensor_type,
                block_size: 32,
                bits_per_weight: 3.0,
                has_scales: true,
                has_min: false,
                has_high_bits: false,
                block_size_bytes: std::mem::size_of::<IQ3_XXSBlock>(),
            },
            TensorType::IQ4_NL | TensorType::IQ4_XS | TensorType::IQ4_UNI => Self {
                tensor_type,
                block_size: 32,
                bits_per_weight: 4.0,
                has_scales: true,
                has_min: false,
                has_high_bits: false,
                block_size_bytes: std::mem::size_of::<IQ4_NLBlock>(),
            },
            // Non-quantized types
            _ => Self {
                tensor_type,
                block_size: 1,
                bits_per_weight: tensor_type.element_size() as f32 * 8.0,
                has_scales: false,
                has_min: false,
                has_high_bits: false,
                block_size_bytes: tensor_type.element_size(),
            },
        }
    }

    /// Calculate the storage size for a given number of elements
    pub fn calculate_storage_size(&self, element_count: u64) -> u64 {
        if self.block_size <= 1 {
            // Non-quantized type
            return element_count * self.block_size_bytes as u64;
        }

        // Block-based quantization
        let num_blocks = (element_count + self.block_size as u64 - 1) / self.block_size as u64;
        num_blocks * self.block_size_bytes as u64
    }

    /// Calculate the number of blocks needed for a given element count
    pub fn calculate_num_blocks(&self, element_count: u64) -> u64 {
        if self.block_size <= 1 {
            return element_count;
        }

        (element_count + self.block_size as u64 - 1) / self.block_size as u64
    }

    /// Check if this quantization format is lossless
    pub fn is_lossless(&self) -> bool {
        // Only non-quantized integer and floating point types are lossless
        matches!(
            self.tensor_type,
            TensorType::F32
                | TensorType::F64
                | TensorType::F16
                | TensorType::BF16
                | TensorType::I8
                | TensorType::I16
                | TensorType::I32
                | TensorType::I64
        )
    }

    /// Get the theoretical dynamic range for this quantization
    pub fn dynamic_range_bits(&self) -> f32 {
        if self.is_lossless() {
            self.bits_per_weight
        } else {
            // Quantized types have reduced dynamic range
            self.bits_per_weight - 1.0 // Reserve 1 bit for sign typically
        }
    }

    /// Estimate the quantization error (higher is worse)
    pub fn quantization_error_estimate(&self) -> f32 {
        if self.is_lossless() {
            0.0
        } else {
            // Simple estimate: error increases exponentially as bits decrease
            2.0_f32.powf(8.0 - self.bits_per_weight)
        }
    }
}

/// Utilities for working with quantization
pub struct QuantizationUtils;

impl QuantizationUtils {
    /// Get all supported quantization types
    pub fn all_quantized_types() -> Vec<TensorType> {
        vec![
            TensorType::Q4_0,
            TensorType::Q4_1,
            TensorType::Q5_0,
            TensorType::Q5_1,
            TensorType::Q8_0,
            TensorType::Q8_1,
            TensorType::Q2_K,
            TensorType::Q3_K,
            TensorType::Q4_K,
            TensorType::Q5_K,
            TensorType::Q6_K,
            TensorType::Q8_K,
            TensorType::IQ2_XXS,
            TensorType::IQ2_XS,
            TensorType::IQ3_XXS,
            TensorType::IQ1_S,
            TensorType::IQ4_NL,
            TensorType::IQ3_S,
            TensorType::IQ2_S,
            TensorType::IQ4_XS,
            TensorType::IQ1_M,
            TensorType::IQ4_UNI,
        ]
    }

    /// Get recommended quantization for model size and quality requirements
    pub fn recommend_quantization(
        model_size_gb: f32,
        target_quality: f32, // 0.0 = maximum compression, 1.0 = maximum quality
    ) -> TensorType {
        // Quality-based selection
        if target_quality > 0.9 {
            return TensorType::F16;
        }

        if target_quality > 0.8 {
            return if model_size_gb > 7.0 { TensorType::Q6_K } else { TensorType::Q8_0 };
        }

        if target_quality > 0.6 {
            return if model_size_gb > 13.0 { TensorType::Q5_K } else { TensorType::Q5_0 };
        }

        if target_quality > 0.4 {
            return if model_size_gb > 13.0 { TensorType::Q4_K } else { TensorType::Q4_0 };
        }

        if target_quality > 0.2 {
            return TensorType::Q3_K;
        }

        // Maximum compression
        if model_size_gb > 30.0 {
            TensorType::IQ2_XS
        } else {
            TensorType::Q2_K
        }
    }

    /// Compare two quantization formats
    pub fn compare_formats(type_a: TensorType, type_b: TensorType) -> std::cmp::Ordering {
        let params_a = QuantizationParams::for_type(type_a);
        let params_b = QuantizationParams::for_type(type_b);

        // Compare by bits per weight (higher is better quality)
        params_a
            .bits_per_weight
            .partial_cmp(&params_b.bits_per_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    }

    /// Get the most similar quantization to a target bit rate
    pub fn find_closest_quantization(target_bits: f32) -> TensorType {
        let all_types = Self::all_quantized_types();
        let mut best_type = TensorType::Q4_0;
        let mut best_diff = f32::INFINITY;

        for tensor_type in all_types {
            let params = QuantizationParams::for_type(tensor_type);
            let diff = (params.bits_per_weight - target_bits).abs();

            if diff < best_diff {
                best_diff = diff;
                best_type = tensor_type;
            }
        }

        best_type
    }

    /// Check if a quantization type is considered modern/recommended
    pub fn is_modern_quantization(tensor_type: TensorType) -> bool {
        matches!(
            tensor_type,
            TensorType::Q4_K
                | TensorType::Q5_K
                | TensorType::Q6_K
                | TensorType::Q8_K
                | TensorType::IQ2_XXS
                | TensorType::IQ2_XS
                | TensorType::IQ3_XXS
                | TensorType::IQ4_NL
                | TensorType::IQ3_S
                | TensorType::IQ2_S
                | TensorType::IQ4_XS
                | TensorType::IQ1_S
                | TensorType::IQ1_M
                | TensorType::IQ4_UNI
        )
    }

    /// Get quantization family (legacy, k-quant, i-quant)
    pub fn get_quantization_family(tensor_type: TensorType) -> &'static str {
        if tensor_type.is_k_quant() {
            "k-quant"
        } else if tensor_type.is_iq_quant() {
            "i-quant"
        } else if tensor_type.is_quantized() {
            "legacy"
        } else {
            "unquantized"
        }
    }
}

impl std::fmt::Display for QuantizationParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (block_size: {}, {:.1} bits/weight, {} bytes/block)",
            self.tensor_type.name(),
            self.block_size,
            self.bits_per_weight,
            self.block_size_bytes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_params_q4_0() {
        let params = QuantizationParams::for_type(TensorType::Q4_0);
        assert_eq!(params.tensor_type, TensorType::Q4_0);
        assert_eq!(params.block_size, 32);
        assert_eq!(params.bits_per_weight, 4.0);
        assert!(params.has_scales);
        assert!(!params.has_min);
        assert!(!params.has_high_bits);
        assert_eq!(params.block_size_bytes, 18); // 2 bytes scale + 16 bytes data
    }

    #[test]
    fn test_quantization_params_q5_1() {
        let params = QuantizationParams::for_type(TensorType::Q5_1);
        assert_eq!(params.block_size, 32);
        assert_eq!(params.bits_per_weight, 5.0);
        assert!(params.has_scales);
        assert!(params.has_min);
        assert!(params.has_high_bits);
    }

    #[test]
    fn test_quantization_params_k_quant() {
        let params = QuantizationParams::for_type(TensorType::Q4_K);
        assert_eq!(params.block_size, 256); // K-quants use larger blocks
        assert_eq!(params.bits_per_weight, 4.0);
    }

    #[test]
    fn test_storage_size_calculation() {
        let params = QuantizationParams::for_type(TensorType::Q4_0);

        // One block worth of elements
        let size_32 = params.calculate_storage_size(32);
        assert_eq!(size_32, params.block_size_bytes as u64);

        // Two blocks worth
        let size_64 = params.calculate_storage_size(64);
        assert_eq!(size_64, 2 * params.block_size_bytes as u64);

        // Partial block (should round up)
        let size_33 = params.calculate_storage_size(33);
        assert_eq!(size_33, 2 * params.block_size_bytes as u64);
    }

    #[test]
    fn test_quantization_properties() {
        let f32_params = QuantizationParams::for_type(TensorType::F32);
        assert!(f32_params.is_lossless());
        assert_eq!(f32_params.quantization_error_estimate(), 0.0);

        let q4_params = QuantizationParams::for_type(TensorType::Q4_0);
        assert!(!q4_params.is_lossless());
        assert!(q4_params.quantization_error_estimate() > 0.0);
    }

    #[test]
    fn test_quantization_utils() {
        let all_quantized = QuantizationUtils::all_quantized_types();
        assert!(!all_quantized.is_empty());
        assert!(all_quantized.contains(&TensorType::Q4_0));
        assert!(all_quantized.contains(&TensorType::Q4_K));
        assert!(!all_quantized.contains(&TensorType::F32));
    }

    #[test]
    fn test_quantization_recommendation() {
        // High quality should prefer less aggressive quantization
        let high_quality = QuantizationUtils::recommend_quantization(7.0, 0.9);
        let params = QuantizationParams::for_type(high_quality);
        assert!(params.bits_per_weight >= 8.0);

        // Low quality should prefer aggressive quantization
        let low_quality = QuantizationUtils::recommend_quantization(7.0, 0.1);
        let params = QuantizationParams::for_type(low_quality);
        assert!(params.bits_per_weight <= 4.0);
    }

    #[test]
    fn test_format_comparison() {
        use std::cmp::Ordering;

        let cmp = QuantizationUtils::compare_formats(TensorType::Q8_0, TensorType::Q4_0);
        assert_eq!(cmp, Ordering::Greater); // Q8_0 has more bits, so it's "greater"

        let cmp = QuantizationUtils::compare_formats(TensorType::Q4_0, TensorType::Q4_K);
        // Both have 4 bits, so should be equal
        assert_eq!(cmp, Ordering::Equal);
    }

    #[test]
    fn test_closest_quantization() {
        let closest_to_4 = QuantizationUtils::find_closest_quantization(4.0);
        let params = QuantizationParams::for_type(closest_to_4);
        assert_eq!(params.bits_per_weight, 4.0);

        let closest_to_6 = QuantizationUtils::find_closest_quantization(6.0);
        let params = QuantizationParams::for_type(closest_to_6);
        assert!((params.bits_per_weight - 6.0).abs() <= 1.0); // Should be close
    }

    #[test]
    fn test_modern_quantization() {
        assert!(QuantizationUtils::is_modern_quantization(TensorType::Q4_K));
        assert!(QuantizationUtils::is_modern_quantization(TensorType::IQ2_XS));
        assert!(!QuantizationUtils::is_modern_quantization(TensorType::Q4_0));
        assert!(!QuantizationUtils::is_modern_quantization(TensorType::F32));
    }

    #[test]
    fn test_quantization_families() {
        assert_eq!(QuantizationUtils::get_quantization_family(TensorType::Q4_0), "legacy");
        assert_eq!(QuantizationUtils::get_quantization_family(TensorType::Q4_K), "k-quant");
        assert_eq!(QuantizationUtils::get_quantization_family(TensorType::IQ2_XS), "i-quant");
        assert_eq!(QuantizationUtils::get_quantization_family(TensorType::F32), "unquantized");
    }

    #[test]
    fn test_block_struct_sizes() {
        use blocks::*;

        // Verify expected block sizes for common formats
        assert_eq!(std::mem::size_of::<Q4_0Block>(), 18);
        assert_eq!(std::mem::size_of::<Q4_1Block>(), 20);
        assert_eq!(std::mem::size_of::<Q5_0Block>(), 22);
        assert_eq!(std::mem::size_of::<Q5_1Block>(), 24);
        assert_eq!(std::mem::size_of::<Q8_0Block>(), 34);
    }

    #[test]
    fn test_params_display() {
        let params = QuantizationParams::for_type(TensorType::Q4_K);
        let display = format!("{}", params);
        assert!(display.contains("Q4_K"));
        assert!(display.contains("256"));
        assert!(display.contains("4.0"));
    }
}
