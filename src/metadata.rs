//! Canonical GGUF metadata types.
//!
//! This module is a convenience re-export of the format-level metadata API.
//! There is one metadata representation throughout the crate, so values built
//! through this path can be validated, serialized, and passed directly to the
//! readers and writers.

pub use crate::format::metadata::{Metadata, MetadataArray, MetadataValue};
