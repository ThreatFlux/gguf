//! High-level builder pattern APIs for creating GGUF files
//!
//! This module provides convenient builder patterns for constructing GGUF files
//! with proper validation and ease of use.

#[cfg(feature = "std")]
pub mod gguf_builder;
pub mod metadata_builder;
pub mod tensor_builder;

#[cfg(feature = "std")]
pub use gguf_builder::*;
pub use metadata_builder::*;
pub use tensor_builder::*;
