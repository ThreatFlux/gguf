//! Tensor type definitions and utilities

#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};
#[cfg(not(feature = "std"))]
use core::fmt;

pub use crate::format::types::GGUFTensorType as TensorType;

/// Extended tensor type information with additional metadata
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct TensorTypeInfo {
    /// The tensor type
    pub tensor_type: TensorType,
    /// Human-readable name
    pub name: String,
    /// Whether this type is quantized
    pub is_quantized: bool,
    /// Whether this type has canonical GGUF/GGML storage geometry
    pub is_supported: bool,
    /// Block size for quantized types (1 for non-quantized)
    pub block_size: usize,
    /// Physical storage bits per weight, including block metadata overhead
    pub bits_per_weight: f32,
    /// Category of quantization
    pub quantization_category: QuantizationCategory,
}

/// Categories of quantization schemes
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationCategory {
    /// No quantization (F32, F16, BF16, etc.)
    None,
    /// Legacy GGML quantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
    Legacy,
    /// K-quant schemes (Q2_K through Q8_K)
    KQuant,
    /// IQ-quant schemes (ultra-low bit)
    IQuant,
    /// Integer types
    Integer,
    /// Ternary quantization types
    Ternary,
    /// Microscaling floating-point quantization types
    Microscaling,
    /// Current small-block quantization types
    Block,
    /// Removed or SDK-only types that cannot be encoded as GGUF tensors
    Unsupported,
}

impl TensorTypeInfo {
    /// Get tensor type information for a given type
    pub fn for_type(tensor_type: TensorType) -> Self {
        let name = tensor_type.name().to_string();
        let is_quantized = tensor_type.is_quantized();
        let block_size = tensor_type.block_size();
        let bits_per_weight = tensor_type.storage_bits_per_weight().unwrap_or(0.0);
        let is_supported = tensor_type.block_size_bytes().is_some();

        let quantization_category = match tensor_type {
            // Floating point types
            TensorType::F32 | TensorType::F64 | TensorType::F16 | TensorType::BF16 => {
                QuantizationCategory::None
            }

            // Integer types
            TensorType::I8 | TensorType::I16 | TensorType::I32 | TensorType::I64 => {
                QuantizationCategory::Integer
            }

            // Legacy quantization
            TensorType::Q4_0
            | TensorType::Q4_1
            | TensorType::Q5_0
            | TensorType::Q5_1
            | TensorType::Q8_0
            | TensorType::Q8_1 => QuantizationCategory::Legacy,

            // K-quant
            TensorType::Q2_K
            | TensorType::Q3_K
            | TensorType::Q4_K
            | TensorType::Q5_K
            | TensorType::Q6_K
            | TensorType::Q8_K => QuantizationCategory::KQuant,

            // IQ-quant (ultra-low bit)
            TensorType::IQ1_S
            | TensorType::IQ1_M
            | TensorType::IQ2_XXS
            | TensorType::IQ2_XS
            | TensorType::IQ2_S
            | TensorType::IQ3_XXS
            | TensorType::IQ3_S
            | TensorType::IQ4_NL
            | TensorType::IQ4_XS => QuantizationCategory::IQuant,

            TensorType::TQ1_0 | TensorType::TQ2_0 => QuantizationCategory::Ternary,
            TensorType::MXFP4 | TensorType::NVFP4 => QuantizationCategory::Microscaling,
            TensorType::Q1_0 | TensorType::Q2_0 => QuantizationCategory::Block,
            TensorType::Q4_2 | TensorType::Q4_3 | TensorType::IQ4_UNI => {
                QuantizationCategory::Unsupported
            }
        };

        Self {
            tensor_type,
            name,
            is_quantized,
            is_supported,
            block_size,
            bits_per_weight,
            quantization_category,
        }
    }

    /// Get the compression ratio compared to F32
    pub fn compression_ratio(&self) -> f32 {
        if self.bits_per_weight > 0.0 {
            32.0 / self.bits_per_weight
        } else {
            0.0
        }
    }

    /// Get the theoretical memory savings compared to F32
    pub fn memory_savings(&self) -> f32 {
        if self.is_supported {
            1.0 - (self.bits_per_weight / 32.0)
        } else {
            0.0
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for TensorTypeInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({:.3} storage bits/weight, {}x compression, {:?})",
            self.name,
            self.bits_per_weight,
            self.compression_ratio(),
            self.quantization_category
        )
    }
}

#[cfg(not(feature = "std"))]
impl fmt::Display for TensorTypeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({:.3} storage bits/weight, {}x compression, {:?})",
            self.name,
            self.bits_per_weight,
            self.compression_ratio(),
            self.quantization_category
        )
    }
}

/// Utility functions for working with tensor types
pub struct TensorTypeUtils;

impl TensorTypeUtils {
    /// Get all supported tensor types
    pub fn all_types() -> Vec<TensorType> {
        vec![
            TensorType::F32,
            TensorType::F16,
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
            TensorType::I8,
            TensorType::I16,
            TensorType::I32,
            TensorType::I64,
            TensorType::F64,
            TensorType::IQ1_M,
            TensorType::BF16,
            TensorType::TQ1_0,
            TensorType::TQ2_0,
            TensorType::MXFP4,
            TensorType::NVFP4,
            TensorType::Q1_0,
            TensorType::Q2_0,
        ]
    }

    /// Get all quantized tensor types
    pub fn quantized_types() -> Vec<TensorType> {
        Self::all_types().into_iter().filter(|t| t.is_quantized()).collect()
    }

    /// Get all non-quantized tensor types
    pub fn non_quantized_types() -> Vec<TensorType> {
        Self::all_types().into_iter().filter(|t| !t.is_quantized()).collect()
    }

    /// Get tensor types by category
    pub fn types_by_category(category: QuantizationCategory) -> Vec<TensorType> {
        Self::all_types()
            .into_iter()
            .filter(|t| {
                let info = TensorTypeInfo::for_type(*t);
                info.quantization_category == category
            })
            .collect()
    }

    /// Check if a tensor type is deprecated
    pub fn is_deprecated(tensor_type: TensorType) -> bool {
        matches!(tensor_type, TensorType::Q4_2 | TensorType::Q4_3)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_type_info() {
        let info = TensorTypeInfo::for_type(TensorType::Q4_0);
        assert_eq!(info.tensor_type, TensorType::Q4_0);
        assert_eq!(info.name, "Q4_0");
        assert!(info.is_quantized);
        assert!(info.is_supported);
        assert_eq!(info.block_size, 32);
        assert_eq!(info.bits_per_weight, 4.5);
        assert_eq!(info.quantization_category, QuantizationCategory::Legacy);
        assert_eq!(info.compression_ratio(), 32.0 / 4.5);
    }

    #[test]
    fn test_tensor_type_info_non_quantized() {
        let info = TensorTypeInfo::for_type(TensorType::F32);
        assert_eq!(info.tensor_type, TensorType::F32);
        assert!(!info.is_quantized);
        assert_eq!(info.block_size, 1);
        assert_eq!(info.bits_per_weight, 32.0);
        assert_eq!(info.quantization_category, QuantizationCategory::None);
        assert_eq!(info.compression_ratio(), 1.0);
    }

    #[test]
    fn test_tensor_type_utils_all_types() {
        let all_types = TensorTypeUtils::all_types();
        assert!(!all_types.is_empty());
        assert!(all_types.contains(&TensorType::F32));
        assert!(all_types.contains(&TensorType::Q4_0));
        assert!(all_types.contains(&TensorType::BF16));
        assert!(all_types.contains(&TensorType::TQ1_0));
        assert!(all_types.contains(&TensorType::TQ2_0));
        assert!(all_types.contains(&TensorType::MXFP4));
        assert!(all_types.contains(&TensorType::NVFP4));
        assert!(all_types.contains(&TensorType::Q1_0));
        assert!(all_types.contains(&TensorType::Q2_0));
        assert!(!all_types.contains(&TensorType::Q4_2));
        assert!(!all_types.contains(&TensorType::IQ4_UNI));
    }

    #[test]
    fn test_tensor_type_utils_categorization() {
        let quantized = TensorTypeUtils::quantized_types();
        let non_quantized = TensorTypeUtils::non_quantized_types();

        assert!(quantized.contains(&TensorType::Q4_0));
        assert!(!quantized.contains(&TensorType::F32));

        assert!(non_quantized.contains(&TensorType::F32));
        assert!(!non_quantized.contains(&TensorType::Q4_0));
    }

    #[test]
    fn test_tensor_type_utils_by_category() {
        let k_quants = TensorTypeUtils::types_by_category(QuantizationCategory::KQuant);
        assert!(k_quants.contains(&TensorType::Q4_K));
        assert!(!k_quants.contains(&TensorType::Q4_0));

        let legacy = TensorTypeUtils::types_by_category(QuantizationCategory::Legacy);
        assert!(legacy.contains(&TensorType::Q4_0));
        assert!(!legacy.contains(&TensorType::Q4_K));

        assert_eq!(
            TensorTypeUtils::types_by_category(QuantizationCategory::Ternary),
            vec![TensorType::TQ1_0, TensorType::TQ2_0]
        );
        assert_eq!(
            TensorTypeUtils::types_by_category(QuantizationCategory::Microscaling),
            vec![TensorType::MXFP4, TensorType::NVFP4]
        );
        assert_eq!(
            TensorTypeUtils::types_by_category(QuantizationCategory::Block),
            vec![TensorType::Q1_0, TensorType::Q2_0]
        );
        assert!(TensorTypeUtils::types_by_category(QuantizationCategory::Unsupported).is_empty());
    }

    #[test]
    fn test_deprecated_types() {
        assert!(TensorTypeUtils::is_deprecated(TensorType::Q4_2));
        assert!(TensorTypeUtils::is_deprecated(TensorType::Q4_3));
        assert!(!TensorTypeUtils::is_deprecated(TensorType::Q4_0));
    }

    #[test]
    fn test_memory_calculations() {
        let q4_info = TensorTypeInfo::for_type(TensorType::Q4_0);
        assert_eq!(q4_info.compression_ratio(), 32.0 / 4.5);
        assert_eq!(q4_info.memory_savings(), 1.0 - 4.5 / 32.0);

        let f32_info = TensorTypeInfo::for_type(TensorType::F32);
        assert_eq!(f32_info.compression_ratio(), 1.0);
        assert_eq!(f32_info.memory_savings(), 0.0);
    }

    #[test]
    fn test_tensor_type_info_display() {
        let info = TensorTypeInfo::for_type(TensorType::Q4_K);
        let display_str = format!("{}", info);
        assert!(display_str.contains("Q4_K"));
        assert!(display_str.contains("4.500 storage"));
        assert!(display_str.contains("KQuant"));
    }

    #[test]
    fn test_unsupported_types_are_not_reported_as_supported_families() {
        for tensor_type in [TensorType::Q4_2, TensorType::Q4_3, TensorType::IQ4_UNI] {
            let info = TensorTypeInfo::for_type(tensor_type);
            assert!(!info.is_supported);
            assert!(!info.is_quantized);
            assert_eq!(info.block_size, 0);
            assert_eq!(info.bits_per_weight, 0.0);
            assert_eq!(info.quantization_category, QuantizationCategory::Unsupported);
            assert_eq!(info.compression_ratio(), 0.0);
        }
    }

    #[test]
    fn test_all_supported_type_info_matches_canonical_geometry() {
        for tensor_type in TensorTypeUtils::all_types() {
            let info = TensorTypeInfo::for_type(tensor_type);
            let block_bytes = tensor_type.block_size_bytes().unwrap();
            assert!(info.is_supported, "{}", tensor_type.name());
            assert_eq!(info.block_size, tensor_type.block_size());
            assert_eq!(
                info.bits_per_weight,
                block_bytes as f32 * 8.0 / info.block_size as f32,
                "{}",
                tensor_type.name()
            );
        }
    }
}
