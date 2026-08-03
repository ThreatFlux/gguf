//! GGUF v3 format structures and raw I/O primitives
//!
//! This module contains the header, metadata, tensor descriptor, type, and
//! alignment structures used by the crate's documented GGUF v3 subset. See
//! the format-support guide for byte-order and tensor-type boundaries.

pub mod alignment;
pub mod constants;
#[cfg(feature = "std")]
pub mod endian;
pub mod header;
pub mod metadata;
pub mod types;

pub use alignment::*;
pub use constants::*;
pub use header::*;
pub use metadata::*;
pub use types::*;
