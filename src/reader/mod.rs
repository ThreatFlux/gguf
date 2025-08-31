//! GGUF file reader functionality
//!
//! This module provides comprehensive support for reading GGUF files,
//! including header parsing, metadata extraction, and tensor data reading.

#[cfg(feature = "std")]
pub mod file_reader;
#[cfg(feature = "std")]
pub mod stream_reader;
#[cfg(feature = "std")]
pub mod tensor_reader;

#[cfg(feature = "std")]
pub use file_reader::*;
#[cfg(feature = "std")]
pub use stream_reader::*;
#[cfg(feature = "std")]
pub use tensor_reader::*;
