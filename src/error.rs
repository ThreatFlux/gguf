//! Error types for the GGUF library

use thiserror::Error;

/// Result type alias for GGUF operations
pub type Result<T> = std::result::Result<T, GGUFError>;

/// Error types that can occur when working with GGUF files
#[derive(Error, Debug)]
pub enum GGUFError {
    /// I/O error occurred
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid GGUF magic number
    #[error("Invalid GGUF magic number: expected 0x{expected:08X}, found 0x{found:08X}")]
    InvalidMagic { expected: u32, found: u32 },

    /// Unsupported GGUF version
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    /// Invalid tensor data
    #[error("Invalid tensor data: {0}")]
    InvalidTensorData(String),

    /// Invalid metadata
    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),

    /// Unexpected end of file
    #[error("Unexpected end of file")]
    UnexpectedEof,

    /// Format error
    #[error("Format error: {0}")]
    Format(String),

    /// Feature not available
    #[error("Feature '{0}' is not available")]
    FeatureUnavailable(String),
}
