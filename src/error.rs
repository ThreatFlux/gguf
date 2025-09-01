//! Error types for the GGUF library

#[cfg(feature = "std")]
use thiserror::Error;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::string::String;

/// Result type alias for GGUF operations
pub type Result<T> = core::result::Result<T, GGUFError>;

/// Error types that can occur when working with GGUF files
#[cfg_attr(feature = "std", derive(Error))]
#[derive(Debug)]
pub enum GGUFError {
    /// I/O error occurred
    #[cfg_attr(feature = "std", error("I/O error: {0}"))]
    #[cfg(feature = "std")]
    Io(#[cfg_attr(feature = "std", from)] std::io::Error),

    /// Invalid GGUF magic number
    #[cfg_attr(
        feature = "std",
        error("Invalid GGUF magic number: expected 0x{expected:08X}, found 0x{found:08X}")
    )]
    InvalidMagic { expected: u32, found: u32 },

    /// Unsupported GGUF version
    #[cfg_attr(feature = "std", error("Unsupported GGUF version: {0}"))]
    UnsupportedVersion(u32),

    /// Invalid tensor data
    #[cfg(any(feature = "std", feature = "alloc"))]
    #[cfg_attr(feature = "std", error("Invalid tensor data: {0}"))]
    InvalidTensorData(String),

    /// Invalid metadata
    #[cfg(any(feature = "std", feature = "alloc"))]
    #[cfg_attr(feature = "std", error("Invalid metadata: {0}"))]
    InvalidMetadata(String),

    /// Unexpected end of file
    #[cfg_attr(feature = "std", error("Unexpected end of file"))]
    UnexpectedEof,

    /// Format error
    #[cfg(any(feature = "std", feature = "alloc"))]
    #[cfg_attr(feature = "std", error("Format error: {0}"))]
    Format(String),

    /// Feature not available
    #[cfg(any(feature = "std", feature = "alloc"))]
    #[cfg_attr(feature = "std", error("Feature '{0}' is not available"))]
    FeatureUnavailable(String),
    
    /// Operation requires allocation but alloc feature is not enabled
    #[cfg_attr(feature = "std", error("Operation requires allocation but 'alloc' feature is not enabled"))]
    AllocationRequired,
}

#[cfg(not(feature = "std"))]
impl core::fmt::Display for GGUFError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GGUFError::InvalidMagic { expected, found } => {
                write!(
                    f,
                    "Invalid GGUF magic number: expected 0x{:08X}, found 0x{:08X}",
                    expected, found
                )
            }
            GGUFError::UnsupportedVersion(v) => write!(f, "Unsupported GGUF version: {}", v),
            #[cfg(any(feature = "std", feature = "alloc"))]
            GGUFError::InvalidTensorData(msg) => write!(f, "Invalid tensor data: {}", msg),
            #[cfg(any(feature = "std", feature = "alloc"))]
            GGUFError::InvalidMetadata(msg) => write!(f, "Invalid metadata: {}", msg),
            GGUFError::UnexpectedEof => write!(f, "Unexpected end of file"),
            #[cfg(any(feature = "std", feature = "alloc"))]
            GGUFError::Format(msg) => write!(f, "Format error: {}", msg),
            #[cfg(any(feature = "std", feature = "alloc"))]
            GGUFError::FeatureUnavailable(feature) => {
                write!(f, "Feature '{}' is not available", feature)
            }
            GGUFError::AllocationRequired => {
                write!(f, "Operation requires allocation but 'alloc' feature is not enabled")
            }
        }
    }
}
