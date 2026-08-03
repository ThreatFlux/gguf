//! Quantization format structures and utilities

use crate::format::types::GGUFTensorType as TensorType;
#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "std")]
use std::cmp;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::vec::Vec;

// Import core modules for no_std compatibility
#[cfg(not(feature = "std"))]
use core::{cmp, fmt};

/// Exact-size opaque views of GGML quantization blocks.
///
/// The encoded byte layouts are stable, but many formats use packed bitfields
/// whose semantic field representation depends on GGML implementation details.
/// Keeping those blocks opaque prevents this crate from promising incorrect
/// field layouts while still making their canonical sizes available.
pub mod blocks {
    macro_rules! opaque_block {
        ($(#[$meta:meta])* $name:ident, $size:expr) => {
            $(#[$meta])*
            #[repr(transparent)]
            #[allow(non_camel_case_types)]
            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            pub struct $name {
                /// The canonical encoded GGML block bytes.
                pub bytes: [u8; $size],
            }

            impl $name {
                /// Encoded size of one block in bytes.
                pub const SIZE: usize = $size;

                /// Construct a block from its exact encoded representation.
                pub const fn from_bytes(bytes: [u8; $size]) -> Self {
                    Self { bytes }
                }

                /// Return the exact encoded representation.
                pub const fn into_bytes(self) -> [u8; $size] {
                    self.bytes
                }
            }
        };
    }

    opaque_block!(/// Q4_0 block: 32 weights in 18 bytes.
        Q4_0Block, 18);
    opaque_block!(/// Q4_1 block: 32 weights in 20 bytes.
        Q4_1Block, 20);
    opaque_block!(/// Q5_0 block: 32 weights in 22 bytes.
        Q5_0Block, 22);
    opaque_block!(/// Q5_1 block: 32 weights in 24 bytes.
        Q5_1Block, 24);
    opaque_block!(/// Q8_0 block: 32 weights in 34 bytes.
        Q8_0Block, 34);
    opaque_block!(/// Q8_1 block: 32 weights in 36 bytes.
        Q8_1Block, 36);
    opaque_block!(/// Q2_K super-block: 256 weights in 84 bytes.
        Q2_KBlock, 84);
    opaque_block!(/// Q3_K super-block: 256 weights in 110 bytes.
        Q3_KBlock, 110);
    opaque_block!(/// Q4_K super-block: 256 weights in 144 bytes.
        Q4_KBlock, 144);
    opaque_block!(/// Q5_K super-block: 256 weights in 176 bytes.
        Q5_KBlock, 176);
    opaque_block!(/// Q6_K super-block: 256 weights in 210 bytes.
        Q6_KBlock, 210);
    opaque_block!(/// Q8_K super-block: 256 weights in 292 bytes.
        Q8_KBlock, 292);
    opaque_block!(/// IQ2_XXS super-block: 256 weights in 66 bytes.
        IQ2_XXSBlock, 66);
    opaque_block!(/// IQ2_XS super-block: 256 weights in 74 bytes.
        IQ2_XSBlock, 74);
    opaque_block!(/// IQ3_XXS super-block: 256 weights in 98 bytes.
        IQ3_XXSBlock, 98);
    opaque_block!(/// IQ1_S super-block: 256 weights in 50 bytes.
        IQ1_SBlock, 50);
    opaque_block!(/// IQ4_NL block: 32 weights in 18 bytes.
        IQ4_NLBlock, 18);
    opaque_block!(/// IQ3_S super-block: 256 weights in 110 bytes.
        IQ3_SBlock, 110);
    opaque_block!(/// IQ2_S super-block: 256 weights in 82 bytes.
        IQ2_SBlock, 82);
    opaque_block!(/// IQ4_XS super-block: 256 weights in 136 bytes.
        IQ4_XSBlock, 136);
    opaque_block!(/// IQ1_M super-block: 256 weights in 56 bytes.
        IQ1_MBlock, 56);
    opaque_block!(/// TQ1_0 block: 256 weights in 54 bytes.
        TQ1_0Block, 54);
    opaque_block!(/// TQ2_0 block: 256 weights in 66 bytes.
        TQ2_0Block, 66);
    opaque_block!(/// MXFP4 block: 32 weights in 17 bytes.
        MXFP4Block, 17);
    opaque_block!(/// NVFP4 block: 64 weights in 36 bytes.
        NVFP4Block, 36);
    opaque_block!(/// Q1_0 block: 128 weights in 18 bytes.
        Q1_0Block, 18);
    opaque_block!(/// Q2_0 block: 64 weights in 18 bytes.
        Q2_0Block, 18);
}

/// Quantization parameters for different tensor types
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
pub struct QuantizationParams {
    /// The tensor type this applies to
    pub tensor_type: TensorType,
    /// Block size (number of elements per quantization block)
    pub block_size: usize,
    /// Physical storage bits per weight, including block metadata overhead
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
        let block_size = tensor_type.block_size();
        let block_size_bytes = tensor_type.block_size_bytes().unwrap_or(0);
        let bits_per_weight = tensor_type.storage_bits_per_weight().unwrap_or(0.0);
        let has_scales = tensor_type.is_quantized();
        let has_min = matches!(
            tensor_type,
            TensorType::Q4_1
                | TensorType::Q5_1
                | TensorType::Q2_K
                | TensorType::Q4_K
                | TensorType::Q5_K
        );
        let has_high_bits = matches!(
            tensor_type,
            TensorType::Q5_0
                | TensorType::Q5_1
                | TensorType::Q3_K
                | TensorType::Q5_K
                | TensorType::Q6_K
        );

        Self {
            tensor_type,
            block_size,
            bits_per_weight,
            has_scales,
            has_min,
            has_high_bits,
            block_size_bytes,
        }
    }

    /// Whether GGUF readers and writers support this tensor type.
    pub fn is_supported(&self) -> bool {
        self.block_size > 0 && self.block_size_bytes > 0
    }

    /// Calculate the storage size for a given number of elements.
    ///
    /// Returns `None` for unsupported tensor types and on arithmetic overflow.
    pub fn calculate_storage_size(&self, element_count: u64) -> Option<u64> {
        self.tensor_type.calculate_size(element_count)
    }

    /// Calculate the number of blocks needed for a given element count
    pub fn calculate_num_blocks(&self, element_count: u64) -> u64 {
        if self.block_size == 0 {
            return 0;
        }
        if self.block_size <= 1 {
            return element_count;
        }

        element_count.div_ceil(self.block_size as u64)
    }
}

/// Utilities for working with quantization
pub struct QuantizationUtils;

impl QuantizationUtils {
    /// Get all supported quantization types
    #[cfg(any(feature = "std", feature = "alloc"))]
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
            TensorType::TQ1_0,
            TensorType::TQ2_0,
            TensorType::MXFP4,
            TensorType::NVFP4,
            TensorType::Q1_0,
            TensorType::Q2_0,
        ]
    }

    /// Compare two quantization formats
    pub fn compare_formats(type_a: TensorType, type_b: TensorType) -> cmp::Ordering {
        let params_a = QuantizationParams::for_type(type_a);
        let params_b = QuantizationParams::for_type(type_b);

        // Compare by physical storage rate. This does not attempt to rank
        // perceptual quality across different quantization families.
        params_a
            .bits_per_weight
            .partial_cmp(&params_b.bits_per_weight)
            .unwrap_or(cmp::Ordering::Equal)
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

    /// Get the storage family for a supported tensor type.
    ///
    /// Removed GGML types and the SDK-only `IQ4_UNI` variant return `None`.
    pub fn get_quantization_family(tensor_type: TensorType) -> Option<&'static str> {
        tensor_type.block_size_bytes()?;
        if matches!(tensor_type, TensorType::TQ1_0 | TensorType::TQ2_0) {
            Some("ternary")
        } else if matches!(tensor_type, TensorType::MXFP4 | TensorType::NVFP4) {
            Some("microscaling")
        } else if matches!(tensor_type, TensorType::Q1_0 | TensorType::Q2_0) {
            Some("block")
        } else if tensor_type.is_k_quant() {
            Some("k-quant")
        } else if tensor_type.is_iq_quant() {
            Some("i-quant")
        } else if tensor_type.is_quantized() {
            Some("legacy")
        } else {
            Some("unquantized")
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for QuantizationParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (block_size: {}, {:.3} storage bits/weight, {} bytes/block)",
            self.tensor_type.name(),
            self.block_size,
            self.bits_per_weight,
            self.block_size_bytes
        )
    }
}

#[cfg(not(feature = "std"))]
impl fmt::Display for QuantizationParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuantizationParams {{ type: {:?}, block_size: {}, block_size_bytes: {}, storage_bits_per_weight: {} }}",
            self.tensor_type,
            self.block_size,
            self.block_size_bytes,
            self.bits_per_weight
        )
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use core::mem;

    const QUANT_GEOMETRY: &[(TensorType, usize, usize)] = &[
        (TensorType::Q4_0, 32, 18),
        (TensorType::Q4_1, 32, 20),
        (TensorType::Q5_0, 32, 22),
        (TensorType::Q5_1, 32, 24),
        (TensorType::Q8_0, 32, 34),
        (TensorType::Q8_1, 32, 36),
        (TensorType::Q2_K, 256, 84),
        (TensorType::Q3_K, 256, 110),
        (TensorType::Q4_K, 256, 144),
        (TensorType::Q5_K, 256, 176),
        (TensorType::Q6_K, 256, 210),
        (TensorType::Q8_K, 256, 292),
        (TensorType::IQ2_XXS, 256, 66),
        (TensorType::IQ2_XS, 256, 74),
        (TensorType::IQ3_XXS, 256, 98),
        (TensorType::IQ1_S, 256, 50),
        (TensorType::IQ4_NL, 32, 18),
        (TensorType::IQ3_S, 256, 110),
        (TensorType::IQ2_S, 256, 82),
        (TensorType::IQ4_XS, 256, 136),
        (TensorType::IQ1_M, 256, 56),
        (TensorType::TQ1_0, 256, 54),
        (TensorType::TQ2_0, 256, 66),
        (TensorType::MXFP4, 32, 17),
        (TensorType::NVFP4, 64, 36),
        (TensorType::Q1_0, 128, 18),
        (TensorType::Q2_0, 64, 18),
    ];

    #[test]
    fn test_quantization_params_q4_0() {
        let params = QuantizationParams::for_type(TensorType::Q4_0);
        assert_eq!(params.tensor_type, TensorType::Q4_0);
        assert_eq!(params.block_size, 32);
        assert_eq!(params.bits_per_weight, 4.5);
        assert!(params.has_scales);
        assert!(!params.has_min);
        assert!(!params.has_high_bits);
        assert_eq!(params.block_size_bytes, 18); // 2 bytes scale + 16 bytes data
    }

    #[test]
    fn test_quantization_params_q5_1() {
        let params = QuantizationParams::for_type(TensorType::Q5_1);
        assert_eq!(params.block_size, 32);
        assert_eq!(params.bits_per_weight, 6.0);
        assert!(params.has_scales);
        assert!(params.has_min);
        assert!(params.has_high_bits);
    }

    #[test]
    fn test_quantization_params_k_quant() {
        let params = QuantizationParams::for_type(TensorType::Q4_K);
        assert_eq!(params.block_size, 256); // K-quants use larger blocks
        assert_eq!(params.bits_per_weight, 4.5);
    }

    #[test]
    fn test_k_quant_minimum_flags() {
        for tensor_type in [TensorType::Q2_K, TensorType::Q4_K, TensorType::Q5_K] {
            assert!(QuantizationParams::for_type(tensor_type).has_min);
        }
        for tensor_type in [TensorType::Q3_K, TensorType::Q6_K, TensorType::Q8_K] {
            assert!(!QuantizationParams::for_type(tensor_type).has_min);
        }
    }

    #[test]
    fn test_storage_size_calculation() {
        let params = QuantizationParams::for_type(TensorType::Q4_0);

        // One block worth of elements
        let size_32 = params.calculate_storage_size(32);
        assert_eq!(size_32, Some(params.block_size_bytes as u64));

        // Two blocks worth
        let size_64 = params.calculate_storage_size(64);
        assert_eq!(size_64, Some(2 * params.block_size_bytes as u64));

        // Partial block (should round up)
        let size_33 = params.calculate_storage_size(33);
        assert_eq!(size_33, Some(2 * params.block_size_bytes as u64));
    }

    #[test]
    fn test_quantization_utils() {
        let all_quantized = QuantizationUtils::all_quantized_types();
        let expected: Vec<_> =
            QUANT_GEOMETRY.iter().map(|(tensor_type, _, _)| *tensor_type).collect();
        assert_eq!(all_quantized, expected);
        assert!(!all_quantized.contains(&TensorType::IQ4_UNI));
        assert!(!all_quantized.contains(&TensorType::Q4_2));
    }

    #[test]
    fn test_format_comparison() {
        use cmp::Ordering;

        let cmp = QuantizationUtils::compare_formats(TensorType::Q8_0, TensorType::Q4_0);
        assert_eq!(cmp, Ordering::Greater); // Q8_0 has more bits, so it's "greater"

        let cmp = QuantizationUtils::compare_formats(TensorType::Q4_0, TensorType::Q4_K);
        // Both have 4 bits, so should be equal
        assert_eq!(cmp, Ordering::Equal);
    }

    #[test]
    fn test_closest_quantization() {
        let closest_to_4_5 = QuantizationUtils::find_closest_quantization(4.5);
        let params = QuantizationParams::for_type(closest_to_4_5);
        assert_eq!(params.bits_per_weight, 4.5);

        let closest_to_6 = QuantizationUtils::find_closest_quantization(6.0);
        let params = QuantizationParams::for_type(closest_to_6);
        assert!((params.bits_per_weight - 6.0).abs() <= 1.0); // Should be close
    }

    #[test]
    fn test_quantization_families() {
        assert_eq!(QuantizationUtils::get_quantization_family(TensorType::Q4_0), Some("legacy"));
        assert_eq!(QuantizationUtils::get_quantization_family(TensorType::Q4_K), Some("k-quant"));
        assert_eq!(QuantizationUtils::get_quantization_family(TensorType::IQ2_XS), Some("i-quant"));
        assert_eq!(QuantizationUtils::get_quantization_family(TensorType::TQ1_0), Some("ternary"));
        assert_eq!(
            QuantizationUtils::get_quantization_family(TensorType::MXFP4),
            Some("microscaling")
        );
        assert_eq!(
            QuantizationUtils::get_quantization_family(TensorType::NVFP4),
            Some("microscaling")
        );
        assert_eq!(QuantizationUtils::get_quantization_family(TensorType::Q1_0), Some("block"));
        assert_eq!(
            QuantizationUtils::get_quantization_family(TensorType::F32),
            Some("unquantized")
        );
        assert_eq!(QuantizationUtils::get_quantization_family(TensorType::IQ4_UNI), None);
    }

    #[test]
    fn test_block_struct_sizes() {
        use blocks::*;

        macro_rules! assert_block_size {
            ($block:ty, $size:expr) => {{
                assert_eq!(mem::size_of::<$block>(), $size);
                assert_eq!(<$block>::SIZE, $size);
            }};
        }

        assert_block_size!(Q4_0Block, 18);
        assert_block_size!(Q4_1Block, 20);
        assert_block_size!(Q5_0Block, 22);
        assert_block_size!(Q5_1Block, 24);
        assert_block_size!(Q8_0Block, 34);
        assert_block_size!(Q8_1Block, 36);
        assert_block_size!(Q2_KBlock, 84);
        assert_block_size!(Q3_KBlock, 110);
        assert_block_size!(Q4_KBlock, 144);
        assert_block_size!(Q5_KBlock, 176);
        assert_block_size!(Q6_KBlock, 210);
        assert_block_size!(Q8_KBlock, 292);
        assert_block_size!(IQ2_XXSBlock, 66);
        assert_block_size!(IQ2_XSBlock, 74);
        assert_block_size!(IQ3_XXSBlock, 98);
        assert_block_size!(IQ1_SBlock, 50);
        assert_block_size!(IQ4_NLBlock, 18);
        assert_block_size!(IQ3_SBlock, 110);
        assert_block_size!(IQ2_SBlock, 82);
        assert_block_size!(IQ4_XSBlock, 136);
        assert_block_size!(IQ1_MBlock, 56);
        assert_block_size!(TQ1_0Block, 54);
        assert_block_size!(TQ2_0Block, 66);
        assert_block_size!(MXFP4Block, 17);
        assert_block_size!(NVFP4Block, 36);
        assert_block_size!(Q1_0Block, 18);
        assert_block_size!(Q2_0Block, 18);
    }

    #[test]
    fn test_all_supported_quantization_geometry_is_consistent() {
        for &(tensor_type, block_size, block_bytes) in QUANT_GEOMETRY {
            let params = QuantizationParams::for_type(tensor_type);
            assert!(params.is_supported(), "{}", tensor_type.name());
            assert!(tensor_type.is_quantized(), "{}", tensor_type.name());
            assert_eq!(tensor_type.element_size(), None, "{}", tensor_type.name());
            assert_eq!(params.block_size, block_size, "{}", tensor_type.name());
            assert_eq!(params.block_size_bytes, block_bytes, "{}", tensor_type.name());
            assert_eq!(tensor_type.calculate_size(block_size as u64), Some(block_bytes as u64));
            let expected_bpw = block_bytes as f32 * 8.0 / block_size as f32;
            assert!((params.bits_per_weight - expected_bpw).abs() < f32::EPSILON);
        }

        for unsupported in [TensorType::Q4_2, TensorType::Q4_3, TensorType::IQ4_UNI] {
            let params = QuantizationParams::for_type(unsupported);
            assert!(!params.is_supported());
            assert_eq!(params.block_size, 0);
            assert_eq!(params.block_size_bytes, 0);
            assert_eq!(params.bits_per_weight, 0.0);
        }
    }

    #[test]
    fn test_params_display() {
        let params = QuantizationParams::for_type(TensorType::Q4_K);
        let display = format!("{}", params);
        assert!(display.contains("Q4_K"));
        assert!(display.contains("256"));
        assert!(display.contains("4.500"));
    }
}
