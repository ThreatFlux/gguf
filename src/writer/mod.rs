//! GGUF file writer functionality
//!
//! This module provides comprehensive support for writing GGUF files,
//! including header writing, metadata serialization, and tensor data writing.

#[cfg(feature = "std")]
pub mod file_writer;
#[cfg(feature = "std")]
pub mod stream_writer;
#[cfg(feature = "std")]
pub mod tensor_writer;

#[cfg(feature = "std")]
pub use file_writer::*;
#[cfg(feature = "std")]
pub use stream_writer::*;
#[cfg(feature = "std")]
pub use tensor_writer::*;
