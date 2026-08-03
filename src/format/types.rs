//! GGUF data types and type system

use crate::error::{GGUFError, Result};
#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::format;

// Import core modules for no_std compatibility
#[cfg(not(feature = "std"))]
use core::fmt;

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

/// Type identifiers for tensor data types in GGUF.
///
/// # 0.3 migration
///
/// Version 0.3 corrected the integer, F64, IQ1_M, and BF16 discriminants to the
/// canonical GGML values: I8=24, I16=25, I32=26, I64=27, F64=28, IQ1_M=29,
/// and BF16=30. Code that persisted the pre-0.3 Rust enum discriminants must
/// migrate those values; the old numbers were not valid GGUF encodings for the
/// named variants. Version 0.3 also recognizes the current TQ1_0=34, TQ2_0=35,
/// MXFP4=39, NVFP4=40, Q1_0=41, and Q2_0=42 raw storage types. `IQ4_UNI`
/// remains only as a source-compatibility variant and is rejected by file
/// readers and writers.
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
    /// IQ2_XXS quantization; canonical geometry is 256 weights per 66-byte block
    IQ2_XXS = 16,
    /// IQ2_XS quantization; canonical geometry is 256 weights per 74-byte block
    IQ2_XS = 17,
    /// IQ3_XXS quantization; canonical geometry is 256 weights per 98-byte block
    IQ3_XXS = 18,
    /// IQ1_S quantization; canonical geometry is 256 weights per 50-byte block
    IQ1_S = 19,
    /// IQ4_NL quantization; canonical geometry is 32 weights per 18-byte block
    IQ4_NL = 20,
    /// IQ3_S quantization; canonical geometry is 256 weights per 110-byte block
    IQ3_S = 21,
    /// IQ2_S quantization; canonical geometry is 256 weights per 82-byte block
    IQ2_S = 22,
    /// IQ4_XS quantization; canonical geometry is 256 weights per 136-byte block
    IQ4_XS = 23,
    /// 8-bit signed integer
    I8 = 24,
    /// 16-bit signed integer
    I16 = 25,
    /// 32-bit signed integer
    I32 = 26,
    /// 64-bit signed integer
    I64 = 27,
    /// 64-bit floating point
    F64 = 28,
    /// IQ1_M quantization; canonical geometry is 256 weights per 56-byte block
    IQ1_M = 29,
    /// bfloat16 (Brain Floating Point)
    BF16 = 30,
    /// TQ1_0 ternary quantization; 256 weights per 54-byte block
    TQ1_0 = 34,
    /// TQ2_0 ternary quantization; 256 weights per 66-byte block
    TQ2_0 = 35,
    /// MXFP4 microscaling quantization; 32 weights per 17-byte block
    MXFP4 = 39,
    /// NVFP4 quantization; 64 weights per 36-byte block
    NVFP4 = 40,
    /// Q1_0 quantization; 128 weights per 18-byte block
    Q1_0 = 41,
    /// Q2_0 quantization; 64 weights per 18-byte block
    Q2_0 = 42,
    /// Legacy SDK-only value retained for source compatibility; not a GGUF tensor type
    IQ4_UNI = u32::MAX,
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
            24 => Ok(GGUFTensorType::I8),
            25 => Ok(GGUFTensorType::I16),
            26 => Ok(GGUFTensorType::I32),
            27 => Ok(GGUFTensorType::I64),
            28 => Ok(GGUFTensorType::F64),
            29 => Ok(GGUFTensorType::IQ1_M),
            30 => Ok(GGUFTensorType::BF16),
            34 => Ok(GGUFTensorType::TQ1_0),
            35 => Ok(GGUFTensorType::TQ2_0),
            39 => Ok(GGUFTensorType::MXFP4),
            40 => Ok(GGUFTensorType::NVFP4),
            41 => Ok(GGUFTensorType::Q1_0),
            42 => Ok(GGUFTensorType::Q2_0),
            _ => Err(GGUFError::Format(format!("Unknown GGUF tensor type: {}", value))),
        }
    }

    /// Get the byte size of one independently addressable scalar value.
    ///
    /// Quantized formats are block-addressed and return `None`; use
    /// [`Self::block_size`] and [`Self::block_size_bytes`] for their geometry.
    pub const fn element_size(self) -> Option<usize> {
        match self {
            GGUFTensorType::F32 | GGUFTensorType::I32 => Some(4),
            GGUFTensorType::F16 | GGUFTensorType::BF16 | GGUFTensorType::I16 => Some(2),
            GGUFTensorType::F64 | GGUFTensorType::I64 => Some(8),
            GGUFTensorType::I8 => Some(1),
            _ => None,
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
            | GGUFTensorType::Q8_K
            | GGUFTensorType::TQ1_0
            | GGUFTensorType::TQ2_0 => 256,
            // IQ super-block types use 256 weights, except IQ4_NL.
            GGUFTensorType::IQ2_XXS
            | GGUFTensorType::IQ2_XS
            | GGUFTensorType::IQ3_XXS
            | GGUFTensorType::IQ1_S
            | GGUFTensorType::IQ3_S
            | GGUFTensorType::IQ2_S
            | GGUFTensorType::IQ4_XS
            | GGUFTensorType::IQ1_M => 256,
            GGUFTensorType::IQ4_NL | GGUFTensorType::MXFP4 => 32,
            GGUFTensorType::NVFP4 | GGUFTensorType::Q2_0 => 64,
            GGUFTensorType::Q1_0 => 128,
            GGUFTensorType::Q4_2 | GGUFTensorType::Q4_3 | GGUFTensorType::IQ4_UNI => 0,
            // Non-quantized types don't have blocks
            _ => 1,
        }
    }

    /// Get the canonical GGML storage size of one block in bytes.
    ///
    /// For scalar types, a block contains one element. Removed GGML types and
    /// the SDK-only `IQ4_UNI` compatibility variant return `None`.
    pub const fn block_size_bytes(self) -> Option<usize> {
        match self {
            GGUFTensorType::F32 | GGUFTensorType::I32 => Some(4),
            GGUFTensorType::F16 | GGUFTensorType::BF16 | GGUFTensorType::I16 => Some(2),
            GGUFTensorType::F64 | GGUFTensorType::I64 => Some(8),
            GGUFTensorType::I8 => Some(1),
            GGUFTensorType::Q4_0 => Some(18),
            GGUFTensorType::Q4_1 => Some(20),
            GGUFTensorType::Q5_0 => Some(22),
            GGUFTensorType::Q5_1 => Some(24),
            GGUFTensorType::Q8_0 => Some(34),
            GGUFTensorType::Q8_1 => Some(36),
            GGUFTensorType::Q2_K => Some(84),
            GGUFTensorType::Q3_K => Some(110),
            GGUFTensorType::Q4_K => Some(144),
            GGUFTensorType::Q5_K => Some(176),
            GGUFTensorType::Q6_K => Some(210),
            GGUFTensorType::Q8_K => Some(292),
            GGUFTensorType::IQ2_XXS => Some(66),
            GGUFTensorType::IQ2_XS => Some(74),
            GGUFTensorType::IQ3_XXS => Some(98),
            GGUFTensorType::IQ1_S => Some(50),
            GGUFTensorType::IQ4_NL => Some(18),
            GGUFTensorType::IQ3_S => Some(110),
            GGUFTensorType::IQ2_S => Some(82),
            GGUFTensorType::IQ4_XS => Some(136),
            GGUFTensorType::IQ1_M => Some(56),
            GGUFTensorType::TQ1_0 => Some(54),
            GGUFTensorType::TQ2_0 => Some(66),
            GGUFTensorType::MXFP4 => Some(17),
            GGUFTensorType::NVFP4 => Some(36),
            GGUFTensorType::Q1_0 | GGUFTensorType::Q2_0 => Some(18),
            GGUFTensorType::Q4_2 | GGUFTensorType::Q4_3 | GGUFTensorType::IQ4_UNI => None,
        }
    }

    /// Return the physical storage bits per weight, including per-block
    /// scales, minima, and other overhead.
    pub fn storage_bits_per_weight(self) -> Option<f32> {
        let elements = self.block_size();
        let bytes = self.block_size_bytes()?;
        if elements == 0 {
            return None;
        }
        Some((bytes as f32 * 8.0) / elements as f32)
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
                | GGUFTensorType::Q4_2
                | GGUFTensorType::Q4_3
                | GGUFTensorType::IQ4_UNI
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
            GGUFTensorType::TQ1_0 => "TQ1_0",
            GGUFTensorType::TQ2_0 => "TQ2_0",
            GGUFTensorType::MXFP4 => "MXFP4",
            GGUFTensorType::NVFP4 => "NVFP4",
            GGUFTensorType::Q1_0 => "Q1_0",
            GGUFTensorType::Q2_0 => "Q2_0",
            GGUFTensorType::IQ4_UNI => "IQ4_UNI",
            GGUFTensorType::I8 => "I8",
            GGUFTensorType::I16 => "I16",
        }
    }

    /// Calculate the storage size in bytes for a given number of elements.
    ///
    /// Returns `None` for removed or non-GGUF tensor types and on arithmetic
    /// overflow.
    pub fn calculate_size(self, element_count: u64) -> Option<u64> {
        self.checked_calculate_size(element_count)
    }

    /// Calculate the storage size using GGML's block layout without overflowing.
    ///
    /// Returns `None` for removed or non-GGUF tensor types and on arithmetic overflow.
    pub fn checked_calculate_size(self, element_count: u64) -> Option<u64> {
        let block_size = u64::try_from(self.block_size()).ok()?;
        if block_size == 0 {
            return None;
        }
        let block_bytes = u64::try_from(self.block_size_bytes()?).ok()?;
        let blocks = element_count.div_ceil(block_size);
        blocks.checked_mul(block_bytes)
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for GGUFValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(not(feature = "std"))]
impl fmt::Display for GGUFValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GGUFValueType::U8 => write!(f, "U8"),
            GGUFValueType::I8 => write!(f, "I8"),
            GGUFValueType::U16 => write!(f, "U16"),
            GGUFValueType::I16 => write!(f, "I16"),
            GGUFValueType::U32 => write!(f, "U32"),
            GGUFValueType::I32 => write!(f, "I32"),
            GGUFValueType::F32 => write!(f, "F32"),
            GGUFValueType::Bool => write!(f, "Bool"),
            GGUFValueType::String => write!(f, "String"),
            GGUFValueType::Array => write!(f, "Array"),
            GGUFValueType::U64 => write!(f, "U64"),
            GGUFValueType::I64 => write!(f, "I64"),
            GGUFValueType::F64 => write!(f, "F64"),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for GGUFTensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(not(feature = "std"))]
impl fmt::Display for GGUFTensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GGUFTensorType::F32 => write!(f, "F32"),
            GGUFTensorType::F16 => write!(f, "F16"),
            GGUFTensorType::Q4_0 => write!(f, "Q4_0"),
            GGUFTensorType::Q4_1 => write!(f, "Q4_1"),
            GGUFTensorType::Q5_0 => write!(f, "Q5_0"),
            GGUFTensorType::Q5_1 => write!(f, "Q5_1"),
            GGUFTensorType::Q8_0 => write!(f, "Q8_0"),
            GGUFTensorType::Q8_1 => write!(f, "Q8_1"),
            GGUFTensorType::Q2_K => write!(f, "Q2_K"),
            GGUFTensorType::Q3_K => write!(f, "Q3_K"),
            GGUFTensorType::Q4_K => write!(f, "Q4_K"),
            GGUFTensorType::Q5_K => write!(f, "Q5_K"),
            GGUFTensorType::Q6_K => write!(f, "Q6_K"),
            GGUFTensorType::Q8_K => write!(f, "Q8_K"),
            GGUFTensorType::I8 => write!(f, "I8"),
            GGUFTensorType::I16 => write!(f, "I16"),
            GGUFTensorType::I32 => write!(f, "I32"),
            GGUFTensorType::I64 => write!(f, "I64"),
            GGUFTensorType::F64 => write!(f, "F64"),
            GGUFTensorType::IQ2_XXS => write!(f, "IQ2_XXS"),
            GGUFTensorType::IQ2_XS => write!(f, "IQ2_XS"),
            GGUFTensorType::IQ3_XXS => write!(f, "IQ3_XXS"),
            GGUFTensorType::IQ1_S => write!(f, "IQ1_S"),
            GGUFTensorType::IQ4_NL => write!(f, "IQ4_NL"),
            GGUFTensorType::IQ3_S => write!(f, "IQ3_S"),
            GGUFTensorType::IQ2_S => write!(f, "IQ2_S"),
            GGUFTensorType::IQ4_XS => write!(f, "IQ4_XS"),
            GGUFTensorType::Q4_2 => write!(f, "Q4_2"),
            GGUFTensorType::Q4_3 => write!(f, "Q4_3"),
            GGUFTensorType::IQ1_M => write!(f, "IQ1_M"),
            GGUFTensorType::BF16 => write!(f, "BF16"),
            GGUFTensorType::TQ1_0 => write!(f, "TQ1_0"),
            GGUFTensorType::TQ2_0 => write!(f, "TQ2_0"),
            GGUFTensorType::MXFP4 => write!(f, "MXFP4"),
            GGUFTensorType::NVFP4 => write!(f, "NVFP4"),
            GGUFTensorType::Q1_0 => write!(f, "Q1_0"),
            GGUFTensorType::Q2_0 => write!(f, "Q2_0"),
            GGUFTensorType::IQ4_UNI => write!(f, "IQ4_UNI"),
        }
    }
}

#[cfg(all(test, feature = "std"))]
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
        assert_eq!(GGUFTensorType::from_u32(30).unwrap(), GGUFTensorType::BF16);
        assert_eq!(GGUFTensorType::I8 as u32, 24);
        assert_eq!(GGUFTensorType::I16 as u32, 25);
        assert_eq!(GGUFTensorType::I32 as u32, 26);
        assert_eq!(GGUFTensorType::I64 as u32, 27);
        assert_eq!(GGUFTensorType::F64 as u32, 28);
        assert_eq!(GGUFTensorType::IQ1_M as u32, 29);
        assert_eq!(GGUFTensorType::from_u32(34).unwrap(), GGUFTensorType::TQ1_0);
        assert_eq!(GGUFTensorType::from_u32(35).unwrap(), GGUFTensorType::TQ2_0);
        assert_eq!(GGUFTensorType::from_u32(39).unwrap(), GGUFTensorType::MXFP4);
        assert_eq!(GGUFTensorType::from_u32(40).unwrap(), GGUFTensorType::NVFP4);
        assert_eq!(GGUFTensorType::from_u32(41).unwrap(), GGUFTensorType::Q1_0);
        assert_eq!(GGUFTensorType::from_u32(42).unwrap(), GGUFTensorType::Q2_0);
        for unsupported in [4, 5, 31, 32, 33, 36, 37, 38, u32::MAX] {
            assert!(GGUFTensorType::from_u32(unsupported).is_err());
        }
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
        assert_eq!(GGUFTensorType::F32.element_size(), Some(4));
        assert_eq!(GGUFTensorType::F16.element_size(), Some(2));
        assert_eq!(GGUFTensorType::Q4_0.element_size(), None);

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
        assert_eq!(GGUFTensorType::F32.calculate_size(100), Some(400));
        assert_eq!(GGUFTensorType::F16.calculate_size(100), Some(200));

        // Quantized types
        let q4_0_size = GGUFTensorType::Q4_0.calculate_size(32); // One block
        assert_eq!(q4_0_size, Some(18));

        let q4_0_size_multi = GGUFTensorType::Q4_0.calculate_size(64); // Two blocks
        assert_eq!(q4_0_size_multi, Some(36));

        assert_eq!(GGUFTensorType::Q2_K.calculate_size(256), Some(84));
        assert_eq!(GGUFTensorType::Q4_K.calculate_size(256), Some(144));
        assert_eq!(GGUFTensorType::IQ2_XXS.calculate_size(256), Some(66));
        assert_eq!(GGUFTensorType::IQ4_NL.calculate_size(32), Some(18));
        assert_eq!(GGUFTensorType::IQ1_M.calculate_size(256), Some(56));
        assert_eq!(GGUFTensorType::TQ1_0.calculate_size(256), Some(54));
        assert_eq!(GGUFTensorType::TQ2_0.calculate_size(256), Some(66));
        assert_eq!(GGUFTensorType::MXFP4.calculate_size(32), Some(17));
        assert_eq!(GGUFTensorType::NVFP4.calculate_size(64), Some(36));
        assert_eq!(GGUFTensorType::Q1_0.calculate_size(128), Some(18));
        assert_eq!(GGUFTensorType::Q2_0.calculate_size(64), Some(18));
        assert_eq!(GGUFTensorType::F64.checked_calculate_size(u64::MAX), None);
        assert_eq!(GGUFTensorType::Q4_2.checked_calculate_size(32), None);
    }
}
