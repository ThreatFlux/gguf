//! Metadata builder utilities

use crate::format::{metadata::MetadataValue, Metadata};

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::string::String;

/// Builder for GGUF metadata
#[derive(Debug, Default, Clone)]
pub struct MetadataBuilder {
    metadata: Metadata,
}

impl MetadataBuilder {
    /// Create a new metadata builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a string value
    pub fn add_string<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), MetadataValue::String(value.into()));
        self
    }

    /// Add a u32 value
    pub fn add_u32<K: Into<String>>(mut self, key: K, value: u32) -> Self {
        self.metadata.insert(key.into(), MetadataValue::U32(value));
        self
    }

    /// Add a u64 value
    pub fn add_u64<K: Into<String>>(mut self, key: K, value: u64) -> Self {
        self.metadata.insert(key.into(), MetadataValue::U64(value));
        self
    }

    /// Add a f32 value
    pub fn add_f32<K: Into<String>>(mut self, key: K, value: f32) -> Self {
        self.metadata.insert(key.into(), MetadataValue::F32(value));
        self
    }

    /// Add a boolean value
    pub fn add_bool<K: Into<String>>(mut self, key: K, value: bool) -> Self {
        self.metadata.insert(key.into(), MetadataValue::Bool(value));
        self
    }

    /// Add any metadata value
    pub fn add<K: Into<String>>(mut self, key: K, value: MetadataValue) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Build the metadata
    pub fn build(self) -> Metadata {
        self.metadata
    }

    /// Get the current size
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }
}

/// Common metadata builders for standard model types
impl MetadataBuilder {
    /// Create metadata for a language model
    pub fn language_model<N: Into<String>>(name: N) -> Self {
        Self::new()
            .add_string("general.architecture", "llama")
            .add_string("general.name", name)
            .add_u32("general.file_type", 1)
    }

    /// Create metadata for a vision model
    pub fn vision_model<N: Into<String>>(name: N) -> Self {
        Self::new()
            .add_string("general.architecture", "clip")
            .add_string("general.name", name)
            .add_u32("general.file_type", 1)
    }

    /// Add common LLaMA parameters
    pub fn with_llama_params(
        self,
        context_length: u64,
        embedding_length: u64,
        vocab_size: u64,
    ) -> Self {
        self.add_u64("llama.context_length", context_length)
            .add_u64("llama.embedding_length", embedding_length)
            .add_u64("llama.vocab_size", vocab_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_builder() {
        let metadata = MetadataBuilder::new()
            .add_string("name", "test")
            .add_u32("version", 1)
            .add_bool("enabled", true)
            .build();

        assert_eq!(metadata.len(), 3);
        assert_eq!(metadata.get_string("name"), Some("test"));
        assert_eq!(metadata.get_u64("version"), Some(1));
        assert_eq!(metadata.get_bool("enabled"), Some(true));
    }

    #[test]
    fn test_language_model_metadata() {
        let metadata = MetadataBuilder::language_model("test_llama")
            .with_llama_params(2048, 4096, 32000)
            .build();

        assert_eq!(metadata.get_string("general.architecture"), Some("llama"));
        assert_eq!(metadata.get_u64("llama.context_length"), Some(2048));
    }
}
