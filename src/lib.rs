//! # GGUF - A Rust library for GGUF file format
//!
//! This library provides support for reading, writing, and manipulating GGUF
//! (GGML Universal Format) files, commonly used for storing large language models.

#![cfg_attr(not(feature = "std"), no_std)]
#![recursion_limit = "8192"]

// Public modules
pub mod builder;
pub mod error;
pub mod format;
pub mod metadata;
pub mod reader;
pub mod tensor;
pub mod writer;

// Optional async support
#[cfg(feature = "async")]
pub mod r#async;

// Optional memory mapping support
#[cfg(feature = "mmap")]
pub mod mmap;

// Re-export main types for convenience
pub use error::{GGUFError, Result};

// Re-export commonly used items in prelude
pub mod prelude {
    #[cfg(feature = "std")]
    pub use crate::builder::gguf_builder::GGUFBuilder;
    pub use crate::error::{GGUFError, Result};
    pub use crate::format::constants::{GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION};
    pub use crate::format::header::GGUFHeader;
    pub use crate::format::metadata::Metadata;
    pub use crate::format::metadata::MetadataValue;
    pub use crate::format::types::GGUFTensorType;
    #[cfg(feature = "std")]
    pub use crate::reader::file_reader::GGUFFileReader;
    #[cfg(feature = "std")]
    pub use crate::writer::file_writer::GGUFFileWriter;
}
