//! GGUF data types and type system

use crate::error::{GGUFError, Result};
#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::format;

/// Type identifiers used in the GGUF format for metadata values
#[repr(u32)]
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GGUFValueType {
    /// 8-bit unsigned integer
    U8 = 0,
    /// 8-bit signed integer
    I8 = 1,
    /// 16-bit unsigned integer
    U16 = 2,
    /// 16-bit signed integer
    I16 = 3,
    /// 32-bit unsigned integer
    U32 = 4,
    /// 32-bit signed integer
    I32 = 5,
    /// 32-bit floating point
    F32 = 6,
    /// Boolean value
    Bool = 7,
    /// UTF-8 string
    String = 8,
    /// Array of values
    Array = 9,
    /// 64-bit unsigned integer
    U64 = 10,
    /// 64-bit signed integer
    I64 = 11,
    /// 64-bit floating point
    F64 = 12,
}

/// Type identifiers for tensor data types in GGUF
#[repr(u32)]
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)] // GGUF spec uses these exact names
pub enum GGUFTensorType {
    /// 32-bit floating point
    F32 = 0,
    /// 16-bit floating point
    F16 = 1,
    /// 4-bit quantized (block size 32)
    Q4_0 = 2,
    /// 4-bit quantized (block size 32, with scales)
    Q4_1 = 3,
    /// 4-bit quantized (superseded)
    Q4_2 = 4,
    /// 4-bit quantized (superseded)
    Q4_3 = 5,
    /// 5-bit quantized (block size 32)
    Q5_0 = 6,
    /// 5-bit quantized (block size 32, with scales)
    Q5_1 = 7,
    /// 8-bit quantized
    Q8_0 = 8,
    /// 8-bit quantized (with scales)
    Q8_1 = 9,
    /// 2-bit quantized (K-quant)
    Q2_K = 10,
    /// 3-bit quantized (K-quant)
    Q3_K = 11,
    /// 4-bit quantized (K-quant)
    Q4_K = 12,
    /// 5-bit quantized (K-quant)
    Q5_K = 13,
    /// 6-bit quantized (K-quant)
    Q6_K = 14,
    /// 8-bit quantized (K-quant)
    Q8_K = 15,
    /// 4-bit quantized (IQ variant)
    IQ2_XXS = 16,
    /// 2-bit quantized (IQ variant)
    IQ2_XS = 17,
    /// 3-bit quantized (IQ variant)
    IQ3_XXS = 18,
    /// 3-bit quantized (IQ variant)
    IQ1_S = 19,
    /// 4-bit quantized (IQ variant)
    IQ4_NL = 20,
    /// 4-bit quantized (IQ variant)
    IQ3_S = 21,
    /// 4-bit quantized (IQ variant)
    IQ2_S = 22,
    /// 4-bit quantized (IQ variant)
    IQ4_XS = 23,
    /// 32-bit signed integer
    I32 = 24,
    /// 64-bit signed integer
    I64 = 25,
    /// 64-bit floating point
    F64 = 26,
    /// 4-bit quantized (IQ variant)
    IQ1_M = 27,
    /// bfloat16 (Brain Floating Point)
    BF16 = 28,
    /// 4-bit quantized (IQ variant, ultra-small)
    IQ4_UNI = 29,
    /// 8-bit signed integer
    I8 = 30,
    /// 16-bit signed integer
    I16 = 31,
}

impl GGUFValueType {
    /// Convert from u32 to GGUFValueType
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(GGUFValueType::U8),
            1 => Ok(GGUFValueType::I8),
            2 => Ok(GGUFValueType::U16),
            3 => Ok(GGUFValueType::I16),
            4 => Ok(GGUFValueType::U32),
            5 => Ok(GGUFValueType::I32),
            6 => Ok(GGUFValueType::F32),
            7 => Ok(GGUFValueType::Bool),
            8 => Ok(GGUFValueType::String),
            9 => Ok(GGUFValueType::Array),
            10 => Ok(GGUFValueType::U64),
            11 => Ok(GGUFValueType::I64),
            12 => Ok(GGUFValueType::F64),
            _ => Err(GGUFError::Format(format!("Unknown GGUF value type: {}", value))),
        }
    }

    /// Get the size in bytes for fixed-size types
    pub fn size_in_bytes(self) -> Option<usize> {
        match self {
            GGUFValueType::U8 | GGUFValueType::I8 | GGUFValueType::Bool => Some(1),
            GGUFValueType::U16 | GGUFValueType::I16 => Some(2),
            GGUFValueType::U32 | GGUFValueType::I32 | GGUFValueType::F32 => Some(4),
            GGUFValueType::U64 | GGUFValueType::I64 | GGUFValueType::F64 => Some(8),
            // Variable-size types
            GGUFValueType::String | GGUFValueType::Array => None,
        }
    }

    /// Check if this type is variable-size
    pub fn is_variable_size(self) -> bool {
        matches!(self, GGUFValueType::String | GGUFValueType::Array)
    }

    /// Check if this type is signed
    pub fn is_signed(self) -> bool {
        matches!(
            self,
            GGUFValueType::I8
                | GGUFValueType::I16
                | GGUFValueType::I32
                | GGUFValueType::I64
                | GGUFValueType::F32
                | GGUFValueType::F64
        )
    }

    /// Check if this type is floating point
    pub fn is_float(self) -> bool {
        matches!(self, GGUFValueType::F32 | GGUFValueType::F64)
    }

    /// Get the alignment requirement for this type
    pub fn alignment(self) -> usize {
        match self {
            GGUFValueType::U8 | GGUFValueType::I8 | GGUFValueType::Bool => 1,
            GGUFValueType::U16 | GGUFValueType::I16 => 2,
            GGUFValueType::U32 | GGUFValueType::I32 | GGUFValueType::F32 => 4,
            GGUFValueType::U64 | GGUFValueType::I64 | GGUFValueType::F64 => 8,
            GGUFValueType::String | GGUFValueType::Array => 1, // No alignment for variable types
        }
    }

    /// Get a human-readable name for the type
    pub fn name(self) -> &'static str {
        match self {
            GGUFValueType::U8 => "u8",
            GGUFValueType::I8 => "i8",
            GGUFValueType::U16 => "u16",
            GGUFValueType::I16 => "i16",
            GGUFValueType::U32 => "u32",
            GGUFValueType::I32 => "i32",
            GGUFValueType::F32 => "f32",
            GGUFValueType::Bool => "bool",
            GGUFValueType::String => "string",
            GGUFValueType::Array => "array",
            GGUFValueType::U64 => "u64",
            GGUFValueType::I64 => "i64",
            GGUFValueType::F64 => "f64",
        }
    }
}

impl GGUFTensorType {
    /// Convert from u32 to GGUFTensorType
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(GGUFTensorType::F32),
            1 => Ok(GGUFTensorType::F16),
            2 => Ok(GGUFTensorType::Q4_0),
            3 => Ok(GGUFTensorType::Q4_1),
            4 => Ok(GGUFTensorType::Q4_2),
            5 => Ok(GGUFTensorType::Q4_3),
            6 => Ok(GGUFTensorType::Q5_0),
            7 => Ok(GGUFTensorType::Q5_1),
            8 => Ok(GGUFTensorType::Q8_0),
            9 => Ok(GGUFTensorType::Q8_1),
            10 => Ok(GGUFTensorType::Q2_K),
            11 => Ok(GGUFTensorType::Q3_K),
            12 => Ok(GGUFTensorType::Q4_K),
            13 => Ok(GGUFTensorType::Q5_K),
            14 => Ok(GGUFTensorType::Q6_K),
            15 => Ok(GGUFTensorType::Q8_K),
            16 => Ok(GGUFTensorType::IQ2_XXS),
            17 => Ok(GGUFTensorType::IQ2_XS),
            18 => Ok(GGUFTensorType::IQ3_XXS),
            19 => Ok(GGUFTensorType::IQ1_S),
            20 => Ok(GGUFTensorType::IQ4_NL),
            21 => Ok(GGUFTensorType::IQ3_S),
            22 => Ok(GGUFTensorType::IQ2_S),
            23 => Ok(GGUFTensorType::IQ4_XS),
            24 => Ok(GGUFTensorType::I32),
            25 => Ok(GGUFTensorType::I64),
            26 => Ok(GGUFTensorType::F64),
            27 => Ok(GGUFTensorType::IQ1_M),
            28 => Ok(GGUFTensorType::BF16),
            29 => Ok(GGUFTensorType::IQ4_UNI),
            30 => Ok(GGUFTensorType::I8),
            31 => Ok(GGUFTensorType::I16),
            _ => Err(GGUFError::Format(format!("Unknown GGUF tensor type: {}", value))),
        }
    }

    /// Get the size in bytes of a single element for this tensor type
    pub fn element_size(self) -> usize {
        match self {
            GGUFTensorType::F32 | GGUFTensorType::I32 => 4,
            GGUFTensorType::F16 | GGUFTensorType::BF16 | GGUFTensorType::I16 => 2,
            GGUFTensorType::F64 | GGUFTensorType::I64 => 8,
            GGUFTensorType::I8 => 1,
            // Quantized types - these are block-based, so element size is not directly applicable
            // Returning 1 as a placeholder, actual size calculations are more complex
            _ => 1,
        }
    }

    /// Get the block size for quantized types
    pub fn block_size(self) -> usize {
        match self {
            GGUFTensorType::Q4_0
            | GGUFTensorType::Q4_1
            | GGUFTensorType::Q5_0
            | GGUFTensorType::Q5_1 => 32,
            GGUFTensorType::Q8_0 | GGUFTensorType::Q8_1 => 32,
            GGUFTensorType::Q2_K
            | GGUFTensorType::Q3_K
            | GGUFTensorType::Q4_K
            | GGUFTensorType::Q5_K
            | GGUFTensorType::Q6_K
            | GGUFTensorType::Q8_K => 256,
            // IQ types typically have smaller block sizes
            GGUFTensorType::IQ2_XXS
            | GGUFTensorType::IQ2_XS
            | GGUFTensorType::IQ3_XXS
            | GGUFTensorType::IQ1_S
            | GGUFTensorType::IQ4_NL
            | GGUFTensorType::IQ3_S
            | GGUFTensorType::IQ2_S
            | GGUFTensorType::IQ4_XS
            | GGUFTensorType::IQ1_M
            | GGUFTensorType::IQ4_UNI => 32,
            // Non-quantized types don't have blocks
            _ => 1,
        }
    }

    /// Check if this tensor type is quantized
    pub fn is_quantized(self) -> bool {
        !matches!(
            self,
            GGUFTensorType::F32
                | GGUFTensorType::F16
                | GGUFTensorType::BF16
                | GGUFTensorType::I32
                | GGUFTensorType::I64
                | GGUFTensorType::F64
                | GGUFTensorType::I8
                | GGUFTensorType::I16
        )
    }

    /// Check if this is a K-quant type
    pub fn is_k_quant(self) -> bool {
        matches!(
            self,
            GGUFTensorType::Q2_K
                | GGUFTensorType::Q3_K
                | GGUFTensorType::Q4_K
                | GGUFTensorType::Q5_K
                | GGUFTensorType::Q6_K
                | GGUFTensorType::Q8_K
        )
    }

    /// Check if this is an IQ-quant type
    pub fn is_iq_quant(self) -> bool {
        matches!(
            self,
            GGUFTensorType::IQ2_XXS
                | GGUFTensorType::IQ2_XS
                | GGUFTensorType::IQ3_XXS
                | GGUFTensorType::IQ1_S
                | GGUFTensorType::IQ4_NL
                | GGUFTensorType::IQ3_S
                | GGUFTensorType::IQ2_S
                | GGUFTensorType::IQ4_XS
                | GGUFTensorType::IQ1_M
                | GGUFTensorType::IQ4_UNI
        )
    }

    /// Get the human-readable name of the tensor type
    pub fn name(self) -> &'static str {
        match self {
            GGUFTensorType::F32 => "F32",
            GGUFTensorType::F16 => "F16",
            GGUFTensorType::Q4_0 => "Q4_0",
            GGUFTensorType::Q4_1 => "Q4_1",
            GGUFTensorType::Q4_2 => "Q4_2",
            GGUFTensorType::Q4_3 => "Q4_3",
            GGUFTensorType::Q5_0 => "Q5_0",
            GGUFTensorType::Q5_1 => "Q5_1",
            GGUFTensorType::Q8_0 => "Q8_0",
            GGUFTensorType::Q8_1 => "Q8_1",
            GGUFTensorType::Q2_K => "Q2_K",
            GGUFTensorType::Q3_K => "Q3_K",
            GGUFTensorType::Q4_K => "Q4_K",
            GGUFTensorType::Q5_K => "Q5_K",
            GGUFTensorType::Q6_K => "Q6_K",
            GGUFTensorType::Q8_K => "Q8_K",
            GGUFTensorType::IQ2_XXS => "IQ2_XXS",
            GGUFTensorType::IQ2_XS => "IQ2_XS",
            GGUFTensorType::IQ3_XXS => "IQ3_XXS",
            GGUFTensorType::IQ1_S => "IQ1_S",
            GGUFTensorType::IQ4_NL => "IQ4_NL",
            GGUFTensorType::IQ3_S => "IQ3_S",
            GGUFTensorType::IQ2_S => "IQ2_S",
            GGUFTensorType::IQ4_XS => "IQ4_XS",
            GGUFTensorType::I32 => "I32",
            GGUFTensorType::I64 => "I64",
            GGUFTensorType::F64 => "F64",
            GGUFTensorType::IQ1_M => "IQ1_M",
            GGUFTensorType::BF16 => "BF16",
            GGUFTensorType::IQ4_UNI => "IQ4_UNI",
            GGUFTensorType::I8 => "I8",
            GGUFTensorType::I16 => "I16",
        }
    }

    /// Calculate the size in bytes for a given number of elements
    pub fn calculate_size(self, element_count: u64) -> u64 {
        if !self.is_quantized() {
            return element_count * self.element_size() as u64;
        }

        let block_size = self.block_size() as u64;
        let num_blocks = element_count.div_ceil(block_size);

        match self {
            GGUFTensorType::Q4_0 => num_blocks * 18, // 2 bytes scale + 16 bytes data per block
            GGUFTensorType::Q4_1 => num_blocks * 20, // 2 bytes scale + 2 bytes min + 16 bytes data per block
            GGUFTensorType::Q5_0 => num_blocks * 22, // 2 bytes scale + 4 bytes high bits + 16 bytes data per block
            GGUFTensorType::Q5_1 => num_blocks * 24, // 2 bytes scale + 2 bytes min + 4 bytes high bits + 16 bytes data per block
            GGUFTensorType::Q8_0 => num_blocks * 34, // 2 bytes scale + 32 bytes data per block
            GGUFTensorType::Q8_1 => num_blocks * 36, // 4 bytes scale + 32 bytes data per block

            // K-quants have more complex layouts
            GGUFTensorType::Q2_K => num_blocks * 82, // Approximate size
            GGUFTensorType::Q3_K => num_blocks * 110, // Approximate size
            GGUFTensorType::Q4_K => num_blocks * 144, // Approximate size
            GGUFTensorType::Q5_K => num_blocks * 176, // Approximate size
            GGUFTensorType::Q6_K => num_blocks * 210, // Approximate size
            GGUFTensorType::Q8_K => num_blocks * 256, // Approximate size

            // IQ types - approximate sizes
            GGUFTensorType::IQ2_XXS => element_count.div_ceil(8) * 2, // ~2 bits per element
            GGUFTensorType::IQ2_XS => element_count.div_ceil(8) * 2,  // ~2 bits per element
            GGUFTensorType::IQ3_XXS => element_count.div_ceil(8) * 3, // ~3 bits per element
            GGUFTensorType::IQ1_S => element_count.div_ceil(8),       // ~1 bit per element
            GGUFTensorType::IQ4_NL => element_count.div_ceil(2),      // ~4 bits per element
            GGUFTensorType::IQ3_S => element_count.div_ceil(8) * 3,   // ~3 bits per element
            GGUFTensorType::IQ2_S => element_count.div_ceil(8) * 2,   // ~2 bits per element
            GGUFTensorType::IQ4_XS => element_count.div_ceil(2),      // ~4 bits per element
            GGUFTensorType::IQ1_M => element_count.div_ceil(8),       // ~1 bit per element
            GGUFTensorType::IQ4_UNI => element_count.div_ceil(2),     // ~4 bits per element

            // Fallback for unsupported types
            _ => element_count * self.element_size() as u64,
        }
    }
}

impl std::fmt::Display for GGUFValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::fmt::Display for GGUFTensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_value_type_conversion() {
        assert_eq!(GGUFValueType::from_u32(0).unwrap(), GGUFValueType::U8);
        assert_eq!(GGUFValueType::from_u32(6).unwrap(), GGUFValueType::F32);
        assert_eq!(GGUFValueType::from_u32(12).unwrap(), GGUFValueType::F64);
        assert!(GGUFValueType::from_u32(99).is_err());
    }

    #[test]
    fn test_gguf_tensor_type_conversion() {
        assert_eq!(GGUFTensorType::from_u32(0).unwrap(), GGUFTensorType::F32);
        assert_eq!(GGUFTensorType::from_u32(2).unwrap(), GGUFTensorType::Q4_0);
        assert_eq!(GGUFTensorType::from_u32(28).unwrap(), GGUFTensorType::BF16);
        assert!(GGUFTensorType::from_u32(99).is_err());
    }

    #[test]
    fn test_value_type_properties() {
        assert_eq!(GGUFValueType::U8.size_in_bytes(), Some(1));
        assert_eq!(GGUFValueType::F32.size_in_bytes(), Some(4));
        assert_eq!(GGUFValueType::String.size_in_bytes(), None);

        assert!(GGUFValueType::String.is_variable_size());
        assert!(!GGUFValueType::U32.is_variable_size());

        assert!(GGUFValueType::I32.is_signed());
        assert!(!GGUFValueType::U32.is_signed());

        assert!(GGUFValueType::F32.is_float());
        assert!(!GGUFValueType::I32.is_float());
    }

    #[test]
    fn test_tensor_type_properties() {
        assert_eq!(GGUFTensorType::F32.element_size(), 4);
        assert_eq!(GGUFTensorType::F16.element_size(), 2);

        assert!(GGUFTensorType::Q4_0.is_quantized());
        assert!(!GGUFTensorType::F32.is_quantized());

        assert!(GGUFTensorType::Q4_K.is_k_quant());
        assert!(!GGUFTensorType::Q4_0.is_k_quant());

        assert!(GGUFTensorType::IQ2_XXS.is_iq_quant());
        assert!(!GGUFTensorType::Q4_0.is_iq_quant());

        assert_eq!(GGUFTensorType::Q4_0.block_size(), 32);
        assert_eq!(GGUFTensorType::Q4_K.block_size(), 256);
    }

    #[test]
    fn test_tensor_size_calculation() {
        // Non-quantized types
        assert_eq!(GGUFTensorType::F32.calculate_size(100), 400);
        assert_eq!(GGUFTensorType::F16.calculate_size(100), 200);

        // Quantized types
        let q4_0_size = GGUFTensorType::Q4_0.calculate_size(32); // One block
        assert_eq!(q4_0_size, 18);

        let q4_0_size_multi = GGUFTensorType::Q4_0.calculate_size(64); // Two blocks
        assert_eq!(q4_0_size_multi, 36);
    }
}
