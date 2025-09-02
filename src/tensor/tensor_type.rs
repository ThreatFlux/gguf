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
use core::{cmp::Ordering, fmt};

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
    /// Block size for quantized types (1 for non-quantized)
    pub block_size: usize,
    /// Bits per weight (approximate for quantized types)
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
}

impl TensorTypeInfo {
    /// Get tensor type information for a given type
    pub fn for_type(tensor_type: TensorType) -> Self {
        let name = tensor_type.name().to_string();
        let is_quantized = tensor_type.is_quantized();
        let block_size = tensor_type.block_size();

        let (bits_per_weight, quantization_category) = match tensor_type {
            // Floating point types
            TensorType::F32 => (32.0, QuantizationCategory::None),
            TensorType::F64 => (64.0, QuantizationCategory::None),
            TensorType::F16 => (16.0, QuantizationCategory::None),
            TensorType::BF16 => (16.0, QuantizationCategory::None),

            // Integer types
            TensorType::I8 => (8.0, QuantizationCategory::Integer),
            TensorType::I16 => (16.0, QuantizationCategory::Integer),
            TensorType::I32 => (32.0, QuantizationCategory::Integer),
            TensorType::I64 => (64.0, QuantizationCategory::Integer),

            // Legacy quantization
            TensorType::Q4_0 | TensorType::Q4_1 => (4.0, QuantizationCategory::Legacy),
            TensorType::Q4_2 | TensorType::Q4_3 => (4.0, QuantizationCategory::Legacy),
            TensorType::Q5_0 | TensorType::Q5_1 => (5.0, QuantizationCategory::Legacy),
            TensorType::Q8_0 | TensorType::Q8_1 => (8.0, QuantizationCategory::Legacy),

            // K-quant
            TensorType::Q2_K => (2.0, QuantizationCategory::KQuant),
            TensorType::Q3_K => (3.0, QuantizationCategory::KQuant),
            TensorType::Q4_K => (4.0, QuantizationCategory::KQuant),
            TensorType::Q5_K => (5.0, QuantizationCategory::KQuant),
            TensorType::Q6_K => (6.0, QuantizationCategory::KQuant),
            TensorType::Q8_K => (8.0, QuantizationCategory::KQuant),

            // IQ-quant (ultra-low bit)
            TensorType::IQ1_S | TensorType::IQ1_M => (1.0, QuantizationCategory::IQuant),
            TensorType::IQ2_XXS | TensorType::IQ2_XS | TensorType::IQ2_S => {
                (2.0, QuantizationCategory::IQuant)
            }
            TensorType::IQ3_XXS | TensorType::IQ3_S => (3.0, QuantizationCategory::IQuant),
            TensorType::IQ4_NL | TensorType::IQ4_XS | TensorType::IQ4_UNI => {
                (4.0, QuantizationCategory::IQuant)
            }
        };

        Self {
            tensor_type,
            name,
            is_quantized,
            block_size,
            bits_per_weight,
            quantization_category,
        }
    }

    /// Check if this tensor type supports specific operations
    pub fn supports_fast_inference(&self) -> bool {
        // Non-quantized types generally support fast inference
        matches!(
            self.quantization_category,
            QuantizationCategory::None | QuantizationCategory::Integer
        )
    }

    /// Check if this tensor type is optimized for memory usage
    pub fn is_memory_optimized(&self) -> bool {
        self.bits_per_weight < 16.0
    }

    /// Get the compression ratio compared to F32
    pub fn compression_ratio(&self) -> f32 {
        32.0 / self.bits_per_weight
    }

    /// Get the theoretical memory savings compared to F32
    pub fn memory_savings(&self) -> f32 {
        1.0 - (self.bits_per_weight / 32.0)
    }

    /// Check if this type is suitable for a given precision requirement
    pub fn meets_precision_requirement(&self, min_bits: f32) -> bool {
        self.bits_per_weight >= min_bits
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for TensorTypeInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({:.1} bits/weight, {}x compression, {:?})",
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
            "{} ({:.1} bits/weight, {}x compression, {:?})",
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
            TensorType::Q4_2,
            TensorType::Q4_3,
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
            TensorType::I32,
            TensorType::I64,
            TensorType::F64,
            TensorType::IQ1_M,
            TensorType::BF16,
            TensorType::IQ4_UNI,
            TensorType::I8,
            TensorType::I16,
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

    /// Find the best tensor type for given requirements
    pub fn find_best_type(
        max_bits: f32,
        prefer_k_quant: bool,
        require_fast_inference: bool,
    ) -> Option<TensorType> {
        let mut candidates: Vec<_> = Self::all_types()
            .into_iter()
            .map(|t| (t, TensorTypeInfo::for_type(t)))
            .filter(|(_, info)| info.bits_per_weight <= max_bits)
            .collect();

        if require_fast_inference {
            candidates.retain(|(_, info)| info.supports_fast_inference());
        }

        if candidates.is_empty() {
            return None;
        }

        // Sort by preference
        candidates.sort_by(|(_, a), (_, b)| {
            // Prefer K-quant if requested
            if prefer_k_quant {
                match (a.quantization_category, b.quantization_category) {
                    (QuantizationCategory::KQuant, QuantizationCategory::KQuant) => {}
                    #[cfg(feature = "std")]
                    (QuantizationCategory::KQuant, _) => return std::cmp::Ordering::Less,
                    #[cfg(feature = "std")]
                    (_, QuantizationCategory::KQuant) => return std::cmp::Ordering::Greater,
                    #[cfg(not(feature = "std"))]
                    (QuantizationCategory::KQuant, _) => return Ordering::Less,
                    #[cfg(not(feature = "std"))]
                    (_, QuantizationCategory::KQuant) => return Ordering::Greater,
                    _ => {}
                }
            }

            // Then prefer higher precision within constraints
            b.bits_per_weight.partial_cmp(&a.bits_per_weight).unwrap()
        });

        candidates.first().map(|(t, _)| *t)
    }

    /// Check if a tensor type is deprecated
    pub fn is_deprecated(tensor_type: TensorType) -> bool {
        matches!(tensor_type, TensorType::Q4_2 | TensorType::Q4_3)
    }

    /// Get the recommended replacement for a deprecated type
    pub fn get_replacement(tensor_type: TensorType) -> Option<TensorType> {
        match tensor_type {
            TensorType::Q4_2 | TensorType::Q4_3 => Some(TensorType::Q4_0),
            _ => None,
        }
    }

    /// Get tensor types suitable for a model size
    pub fn types_for_model_size(model_size_gb: f32) -> Vec<TensorType> {
        let mut types = Vec::new();

        // For small models, all types are fine
        if model_size_gb < 1.0 {
            return Self::all_types();
        }

        // For medium models, prefer more aggressive quantization
        if model_size_gb < 7.0 {
            types.extend_from_slice(&[
                TensorType::Q4_0,
                TensorType::Q4_1,
                TensorType::Q5_0,
                TensorType::Q5_1,
                TensorType::Q4_K,
                TensorType::Q5_K,
                TensorType::Q6_K,
                TensorType::F16,
                TensorType::BF16,
            ]);
        }

        // For large models, prefer very aggressive quantization
        if model_size_gb < 30.0 {
            types.extend_from_slice(&[
                TensorType::Q2_K,
                TensorType::Q3_K,
                TensorType::Q4_K,
                TensorType::IQ2_XS,
                TensorType::IQ3_S,
                TensorType::IQ4_NL,
            ]);
        } else {
            // For very large models, use the most aggressive quantization
            types.extend_from_slice(&[
                TensorType::IQ1_S,
                TensorType::IQ1_M,
                TensorType::IQ2_XXS,
                TensorType::IQ2_XS,
                TensorType::IQ3_XXS,
            ]);
        }

        types
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
        assert_eq!(info.block_size, 32);
        assert_eq!(info.bits_per_weight, 4.0);
        assert_eq!(info.quantization_category, QuantizationCategory::Legacy);
        assert_eq!(info.compression_ratio(), 8.0);
        assert!(info.is_memory_optimized());
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
        assert!(info.supports_fast_inference());
    }

    #[test]
    fn test_tensor_type_utils_all_types() {
        let all_types = TensorTypeUtils::all_types();
        assert!(!all_types.is_empty());
        assert!(all_types.contains(&TensorType::F32));
        assert!(all_types.contains(&TensorType::Q4_0));
        assert!(all_types.contains(&TensorType::BF16));
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
    }

    #[test]
    fn test_find_best_type() {
        // Test with high precision requirement
        let best = TensorTypeUtils::find_best_type(8.0, false, false);
        assert!(best.is_some());
        let info = TensorTypeInfo::for_type(best.unwrap());
        assert!(info.bits_per_weight <= 8.0);

        // Test with K-quant preference
        let best_k = TensorTypeUtils::find_best_type(8.0, true, false);
        if let Some(best_type) = best_k {
            let info = TensorTypeInfo::for_type(best_type);
            if info.bits_per_weight <= 8.0 {
                // Should prefer K-quant if available in range
                let k_quants = TensorTypeUtils::types_by_category(QuantizationCategory::KQuant);
                let has_k_in_range = k_quants.iter().any(|t| {
                    let t_info = TensorTypeInfo::for_type(*t);
                    t_info.bits_per_weight <= 8.0
                });
                if has_k_in_range {
                    assert_eq!(info.quantization_category, QuantizationCategory::KQuant);
                }
            }
        }
    }

    #[test]
    fn test_deprecated_types() {
        assert!(TensorTypeUtils::is_deprecated(TensorType::Q4_2));
        assert!(TensorTypeUtils::is_deprecated(TensorType::Q4_3));
        assert!(!TensorTypeUtils::is_deprecated(TensorType::Q4_0));

        assert_eq!(TensorTypeUtils::get_replacement(TensorType::Q4_2), Some(TensorType::Q4_0));
        assert_eq!(TensorTypeUtils::get_replacement(TensorType::Q4_0), None);
    }

    #[test]
    fn test_types_for_model_size() {
        let small_model_types = TensorTypeUtils::types_for_model_size(0.5);
        let large_model_types = TensorTypeUtils::types_for_model_size(50.0);

        // Small models should have more type options
        assert!(small_model_types.len() >= large_model_types.len());

        // Large models should prefer aggressive quantization
        assert!(large_model_types.iter().any(|t| {
            let info = TensorTypeInfo::for_type(*t);
            info.bits_per_weight < 2.5
        }));
    }

    #[test]
    fn test_precision_requirements() {
        let info = TensorTypeInfo::for_type(TensorType::Q4_0);
        assert!(info.meets_precision_requirement(4.0));
        assert!(!info.meets_precision_requirement(8.0));

        let f32_info = TensorTypeInfo::for_type(TensorType::F32);
        assert!(f32_info.meets_precision_requirement(16.0));
    }

    #[test]
    fn test_memory_calculations() {
        let q4_info = TensorTypeInfo::for_type(TensorType::Q4_0);
        assert_eq!(q4_info.compression_ratio(), 8.0);
        assert_eq!(q4_info.memory_savings(), 0.875);

        let f32_info = TensorTypeInfo::for_type(TensorType::F32);
        assert_eq!(f32_info.compression_ratio(), 1.0);
        assert_eq!(f32_info.memory_savings(), 0.0);
    }

    #[test]
    fn test_tensor_type_info_display() {
        let info = TensorTypeInfo::for_type(TensorType::Q4_K);
        let display_str = format!("{}", info);
        assert!(display_str.contains("Q4_K"));
        assert!(display_str.contains("4.0 bits"));
        assert!(display_str.contains("KQuant"));
    }
}
