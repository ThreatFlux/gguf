//! # GGUF - A Rust library for GGUF file format
//!
//! This library provides support for reading, writing, and manipulating GGUF
//! (GGML Universal Format) files, commonly used for storing large language models.
//!
//! ## Basic Usage
//!
//! ### Creating a GGUF file
//!
//! ```rust
//! # use gguf_rs::prelude::*;
//! # use gguf_rs::format::metadata::MetadataValue;
//! # use gguf_rs::tensor::TensorType;
//! # fn main() -> Result<()> {
//! // Create a simple GGUF file
//! let builder = GGUFBuilder::simple("my_model", "A test model")
//!     .add_metadata("version", MetadataValue::String("1.0".to_string()))
//!     .add_f32_tensor("weights", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
//!
//! let (bytes, result) = builder.build_to_bytes()?;
//! println!("Created GGUF file with {} bytes", bytes.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Reading a GGUF file
//!
//! ```rust
//! # use gguf_rs::prelude::*;
//! # use std::io::Cursor;
//! # fn example_data() -> Vec<u8> {
//! #     // Create minimal valid GGUF data for testing
//! #     use gguf_rs::format::constants::*;
//! #     let mut data = Vec::new();
//! #     // Header: magic, version, tensor_count, metadata_count
//! #     data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
//! #     data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
//! #     data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
//! #     data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata entry
//! #     // Metadata: key "test", value "value" (string type)
//! #     data.extend_from_slice(&4u64.to_le_bytes()); // key length
//! #     data.extend_from_slice(b"test"); // key
//! #     data.extend_from_slice(&8u32.to_le_bytes()); // string type
//! #     data.extend_from_slice(&5u64.to_le_bytes()); // value length
//! #     data.extend_from_slice(b"value"); // value
//! #     // Align to 32 bytes for tensor data (none in this case)
//! #     while data.len() % 32 != 0 { data.push(0); }
//! #     data
//! # }
//! # fn main() -> Result<()> {
//! let data = example_data();
//! let reader = GGUFFileReader::new(Cursor::new(data))?;
//!
//! println!("GGUF version: {}", reader.header().version);
//! println!("Metadata entries: {}", reader.metadata().len());
//! println!("Tensors: {}", reader.tensor_count());
//! # Ok(())
//! # }
//! ```

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
