//! Metadata handling for GGUF files

pub use crate::format::metadata::MetadataValue;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::string::String;

/// Metadata collection for a GGUF file
#[cfg(any(feature = "std", feature = "alloc"))]
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    /// Key-value pairs of metadata
    pub entries: HashMap<String, MetadataValue>,
}

/// Metadata collection for a GGUF file (no_std + no_alloc variant)
#[cfg(not(any(feature = "std", feature = "alloc")))]
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    // Placeholder for no_std + no_alloc builds
}

#[cfg(any(feature = "std", feature = "alloc"))]
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

#[cfg(not(any(feature = "std", feature = "alloc")))]
impl Metadata {
    /// Create new empty metadata (no-op for no_std + no_alloc)
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a metadata entry (no-op for no_std + no_alloc)
    pub fn insert<K>(&mut self, _key: K, _value: MetadataValue) {
        // No-op: can't store data without allocation
    }

    /// Get a metadata value (always returns None for no_std + no_alloc)
    pub fn get(&self, _key: &str) -> Option<&MetadataValue> {
        None
    }

    /// Get the number of entries (always 0 for no_std + no_alloc)
    pub fn len(&self) -> usize {
        0
    }

    /// Check if empty (always true for no_std + no_alloc)
    pub fn is_empty(&self) -> bool {
        true
    }
}
