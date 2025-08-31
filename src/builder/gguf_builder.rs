//! High-level GGUF file builder

use crate::error::{GGUFError, Result};
use crate::format::Metadata;
use crate::tensor::{TensorData, TensorInfo, TensorShape, TensorType};
use crate::writer::{GGUFFileWriter, GGUFWriteResult, GGUFWriterConfig};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// High-level builder for creating GGUF files
#[derive(Debug, Default)]
pub struct GGUFBuilder {
    /// Metadata for the file
    metadata: Metadata,
    /// Tensors to include
    tensors: Vec<(TensorInfo, TensorData)>,
    /// Writer configuration
    config: Option<GGUFWriterConfig>,
}

impl GGUFBuilder {
    /// Create a new GGUF builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add metadata key-value pair
    pub fn add_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<crate::format::metadata::MetadataValue>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add a tensor with data
    pub fn add_tensor<N>(
        mut self,
        name: N,
        shape: Vec<u64>,
        tensor_type: TensorType,
        data: Vec<u8>,
    ) -> Result<Self>
    where
        N: Into<String>,
    {
        let shape = TensorShape::new(shape)?;
        let tensor_info = TensorInfo::new(name.into(), shape, tensor_type, 0);
        let tensor_data = TensorData::new_owned(data);

        // Validate tensor data size
        if tensor_data.len() != tensor_info.expected_data_size() as usize {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor data size mismatch: expected {}, got {}",
                tensor_info.expected_data_size(),
                tensor_data.len()
            )));
        }

        self.tensors.push((tensor_info, tensor_data));
        Ok(self)
    }

    /// Add a tensor with TensorData
    pub fn add_tensor_with_data<N>(
        mut self,
        name: N,
        shape: Vec<u64>,
        tensor_type: TensorType,
        data: TensorData,
    ) -> Result<Self>
    where
        N: Into<String>,
    {
        let shape = TensorShape::new(shape)?;
        let tensor_info = TensorInfo::new(name.into(), shape, tensor_type, 0);

        // Validate tensor data size
        if data.len() != tensor_info.expected_data_size() as usize {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor data size mismatch: expected {}, got {}",
                tensor_info.expected_data_size(),
                data.len()
            )));
        }

        self.tensors.push((tensor_info, data));
        Ok(self)
    }

    /// Set writer configuration
    pub fn with_config(mut self, config: GGUFWriterConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set tensor alignment
    pub fn with_tensor_alignment(mut self, alignment: usize) -> Self {
        let mut config = self.config.unwrap_or_default();
        config.tensor_alignment = alignment;
        self.config = Some(config);
        self
    }

    /// Enable data validation
    pub fn with_validation(mut self, validate: bool) -> Self {
        let mut config = self.config.unwrap_or_default();
        config.validate_data = validate;
        self.config = Some(config);
        self
    }

    /// Get the number of tensors
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Get the number of metadata entries
    pub fn metadata_count(&self) -> usize {
        self.metadata.len()
    }

    /// Calculate total tensor data size
    pub fn total_tensor_size(&self) -> u64 {
        self.tensors.iter().map(|(info, _)| info.expected_data_size()).sum()
    }

    /// Get tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|(info, _)| info.name()).collect()
    }

    /// Check if a tensor exists
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.iter().any(|(info, _)| info.name() == name)
    }

    /// Remove a tensor by name
    pub fn remove_tensor(mut self, name: &str) -> Self {
        self.tensors.retain(|(info, _)| info.name() != name);
        self
    }

    /// Clear all tensors
    pub fn clear_tensors(mut self) -> Self {
        self.tensors.clear();
        self
    }

    /// Clear all metadata
    pub fn clear_metadata(mut self) -> Self {
        self.metadata = Metadata::new();
        self
    }

    /// Build and write to a writer
    pub fn build_to_writer<W: Write>(self, writer: W) -> Result<GGUFWriteResult> {
        let config = self.config.unwrap_or_default();
        let mut gguf_writer = GGUFFileWriter::with_config(writer, config);

        gguf_writer.write_complete_file(&self.metadata, &self.tensors)
    }

    /// Build and write to a file path
    pub fn build_to_file<P: AsRef<Path>>(self, path: P) -> Result<GGUFWriteResult> {
        let file = File::create(path)?;
        let buf_writer = BufWriter::new(file);
        self.build_to_writer(buf_writer)
    }

    /// Build and return as bytes
    pub fn build_to_bytes(self) -> Result<(Vec<u8>, GGUFWriteResult)> {
        let mut buffer = Vec::new();
        let result = self.build_to_writer(&mut buffer)?;
        Ok((buffer, result))
    }

    /// Validate the builder state before building
    pub fn validate(&self) -> Result<()> {
        // Check for duplicate tensor names
        let mut names = std::collections::HashSet::new();
        for (tensor_info, _) in &self.tensors {
            if !names.insert(tensor_info.name()) {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Duplicate tensor name: '{}'",
                    tensor_info.name()
                )));
            }
        }

        // Validate each tensor
        for (tensor_info, tensor_data) in &self.tensors {
            tensor_info.validate()?;
            tensor_data.validate()?;
        }

        Ok(())
    }

    /// Create a summary of what will be built
    pub fn summary(&self) -> GGUFBuilderSummary {
        let tensor_types = self.tensors.iter().fold(HashMap::new(), |mut acc, (info, _)| {
            *acc.entry(info.tensor_type()).or_insert(0) += 1;
            acc
        });

        GGUFBuilderSummary {
            tensor_count: self.tensors.len(),
            metadata_count: self.metadata.len(),
            total_tensor_size: self.total_tensor_size(),
            tensor_types,
            tensor_names: self.tensor_names().iter().map(|&s| s.to_string()).collect(),
        }
    }
}

/// Summary of what a GGUFBuilder will create
#[derive(Debug, Clone)]
pub struct GGUFBuilderSummary {
    /// Number of tensors
    pub tensor_count: usize,
    /// Number of metadata entries
    pub metadata_count: usize,
    /// Total size of tensor data
    pub total_tensor_size: u64,
    /// Count of each tensor type
    pub tensor_types: HashMap<TensorType, usize>,
    /// List of tensor names
    pub tensor_names: Vec<String>,
}

/// Convenience functions for common GGUF creation patterns
impl GGUFBuilder {
    /// Create a simple GGUF file with basic metadata
    pub fn simple<N, M>(name: N, model_name: M) -> Self
    where
        N: Into<String>,
        M: Into<String>,
    {
        use crate::format::metadata::MetadataValue;

        Self::new()
            .add_metadata("general.name", MetadataValue::String(name.into()))
            .add_metadata("general.description", MetadataValue::String(model_name.into()))
            .add_metadata("general.file_type", MetadataValue::U32(1))
    }

    /// Create a GGUF builder for a language model
    pub fn language_model<N>(name: N, context_length: u64, embedding_size: u64) -> Self
    where
        N: Into<String>,
    {
        use crate::format::metadata::MetadataValue;

        Self::simple(name, "Language Model")
            .add_metadata("llama.context_length", MetadataValue::U64(context_length))
            .add_metadata("llama.embedding_length", MetadataValue::U64(embedding_size))
            .add_metadata("general.architecture", MetadataValue::String("llama".to_string()))
    }

    /// Add a vocabulary tensor (common for language models)
    pub fn add_vocabulary(
        self,
        vocab_size: u64,
        embedding_size: u64,
        data: Vec<u8>,
    ) -> Result<Self> {
        self.add_tensor(
            "token_embd.weight",
            vec![vocab_size, embedding_size],
            TensorType::F32,
            data,
        )
    }

    /// Add an output projection tensor
    pub fn add_output_projection(
        self,
        vocab_size: u64,
        embedding_size: u64,
        data: Vec<u8>,
    ) -> Result<Self> {
        self.add_tensor("output.weight", vec![embedding_size, vocab_size], TensorType::F32, data)
    }
}

impl std::fmt::Display for GGUFBuilderSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GGUF Builder Summary:")?;
        writeln!(f, "  Tensors: {}", self.tensor_count)?;
        writeln!(f, "  Metadata entries: {}", self.metadata_count)?;
        writeln!(f, "  Total tensor size: {} bytes", self.total_tensor_size)?;
        writeln!(f, "  Tensor types:")?;
        for (tensor_type, count) in &self.tensor_types {
            writeln!(f, "    {}: {}", tensor_type.name(), count)?;
        }
        writeln!(f, "  Tensor names: {:?}", self.tensor_names)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::metadata::MetadataValue;

    #[test]
    fn test_gguf_builder_creation() {
        let builder = GGUFBuilder::new();
        assert_eq!(builder.tensor_count(), 0);
        assert_eq!(builder.metadata_count(), 0);
    }

    #[test]
    fn test_add_metadata() {
        let builder = GGUFBuilder::new()
            .add_metadata("test_key", MetadataValue::String("test_value".to_string()))
            .add_metadata("number", MetadataValue::U32(42));

        assert_eq!(builder.metadata_count(), 2);
    }

    #[test]
    fn test_add_tensor() {
        let builder = GGUFBuilder::new();
        let data = vec![0u8; 16]; // 4 F32 values

        let builder = builder.add_tensor("test_tensor", vec![2, 2], TensorType::F32, data).unwrap();

        assert_eq!(builder.tensor_count(), 1);
        assert!(builder.has_tensor("test_tensor"));
        assert_eq!(builder.total_tensor_size(), 16);
    }

    #[test]
    fn test_tensor_validation() {
        let builder = GGUFBuilder::new();
        let wrong_size_data = vec![0u8; 8]; // Should be 16 for 2x2 F32

        let result = builder.add_tensor("test", vec![2, 2], TensorType::F32, wrong_size_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_validation() {
        let mut builder = GGUFBuilder::new();

        // Add duplicate tensor names
        let data = vec![0u8; 4];
        builder = builder.add_tensor("dup", vec![1], TensorType::F32, data.clone()).unwrap();
        builder = builder.add_tensor("dup", vec![1], TensorType::F32, data).unwrap();

        let result = builder.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_builder() {
        let builder = GGUFBuilder::simple("test_model", "A test model");
        assert!(builder.metadata_count() > 0);
    }

    #[test]
    fn test_language_model_builder() {
        let builder = GGUFBuilder::language_model("llama_test", 2048, 4096);
        assert!(builder.metadata_count() > 0);

        let summary = builder.summary();
        assert_eq!(summary.tensor_count, 0);
        assert!(summary.metadata_count > 0);
    }

    #[test]
    fn test_build_to_bytes() {
        let builder = GGUFBuilder::simple("test", "test")
            .add_tensor("small_tensor", vec![2], TensorType::F32, vec![0u8; 8])
            .unwrap();

        let (bytes, result) = builder.build_to_bytes().unwrap();
        assert!(!bytes.is_empty());
        assert!(result.total_bytes_written > 0);
    }

    #[test]
    fn test_tensor_operations() {
        let builder = GGUFBuilder::new()
            .add_tensor("tensor1", vec![2], TensorType::F32, vec![0u8; 8])
            .unwrap()
            .add_tensor("tensor2", vec![3], TensorType::F32, vec![0u8; 12])
            .unwrap();

        assert_eq!(builder.tensor_count(), 2);

        let builder = builder.remove_tensor("tensor1");
        assert_eq!(builder.tensor_count(), 1);
        assert!(!builder.has_tensor("tensor1"));
        assert!(builder.has_tensor("tensor2"));

        let builder = builder.clear_tensors();
        assert_eq!(builder.tensor_count(), 0);
    }

    #[test]
    fn test_summary_display() {
        let builder = GGUFBuilder::simple("test", "test")
            .add_tensor("t1", vec![2], TensorType::F32, vec![0u8; 8])
            .unwrap()
            .add_tensor("t2", vec![1], TensorType::F16, vec![0u8; 2])
            .unwrap();

        let summary = builder.summary();
        let display = format!("{}", summary);

        assert!(display.contains("Tensors: 2"));
        assert!(display.contains("F32"));
        assert!(display.contains("F16"));
    }
}
