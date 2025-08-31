//! Metadata handling for GGUF files

pub use crate::format::metadata::MetadataValue;
use std::collections::HashMap;

/// Metadata collection for a GGUF file
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    /// Key-value pairs of metadata
    pub entries: HashMap<String, MetadataValue>,
}

impl Metadata {
    /// Create new empty metadata
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a metadata entry
    pub fn insert(&mut self, key: String, value: MetadataValue) {
        self.entries.insert(key, value);
    }

    /// Get a metadata value
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
