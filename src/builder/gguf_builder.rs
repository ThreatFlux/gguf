//! High-level GGUF file builder
//!
//! This module provides a high-level builder pattern for creating GGUF files.
//!
//! ## Example
//!
//! ```rust
//! # use gguf_rs_lib::prelude::*;
//! # fn main() -> Result<()> {
//! // Create a language model GGUF file
//! let builder = GGUFBuilder::language_model("my_llm", 2048, 768)
//!     .add_f32_tensor("token_embd.weight", vec![768, 1000], vec![0.0; 768_000])?;
//!
//! let (bytes, result) = builder.build_to_bytes()?;
//! println!("Built GGUF file: {} bytes", result.total_bytes_written);
//! # Ok(())
//! # }
//! ```

use crate::error::{GGUFError, Result};
use crate::format::constants::GGUF_QUANTIZATION_VERSION;
use crate::format::Metadata;
use crate::tensor::{TensorData, TensorInfo, TensorShape, TensorType};

#[cfg(feature = "std")]
use crate::writer::{
    create_gguf_file_with_config, GGUFFileWriter, GGUFWriteResult, GGUFWriterConfig,
};
#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "std")]
use std::io::Write;
#[cfg(feature = "std")]
use std::path::Path;

/// High-level builder for creating GGUF files
#[derive(Debug, Default)]
pub struct GGUFBuilder {
    /// Metadata for the file
    metadata: Metadata,
    /// Tensors to include
    tensors: Vec<(TensorInfo, TensorData)>,
    /// Tensor names used for immediate duplicate detection
    tensor_name_index: HashSet<String>,
    /// Writer configuration
    config: Option<GGUFWriterConfig>,
}

impl GGUFBuilder {
    /// Create a new GGUF builder
    ///
    /// # Example
    ///
    /// ```rust
    /// # use gguf_rs_lib::prelude::*;
    /// let builder = GGUFBuilder::new();
    /// assert_eq!(builder.tensor_count(), 0);
    /// assert_eq!(builder.metadata_count(), 0);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```rust
    /// # use gguf_rs_lib::prelude::*;
    /// # use gguf_rs_lib::tensor::TensorType;
    /// # fn main() -> Result<()> {
    /// let builder = GGUFBuilder::new()
    ///     .add_tensor("weights", vec![2, 3], TensorType::F32, vec![0u8; 24])?;
    ///
    /// assert_eq!(builder.tensor_count(), 1);
    /// assert!(builder.has_tensor("weights"));
    /// # Ok(())
    /// # }
    /// ```
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
        self.try_push_tensor(name.into(), shape, tensor_type, TensorData::new_owned(data))?;
        Ok(self)
    }

    fn try_push_tensor(
        &mut self,
        name: String,
        shape: Vec<u64>,
        tensor_type: TensorType,
        tensor_data: TensorData,
    ) -> Result<()> {
        let shape = TensorShape::new(shape)?;
        let tensor_info = TensorInfo::new(name, shape, tensor_type, 0);
        tensor_info.validate()?;

        // Validate tensor data size
        let expected_size =
            usize::try_from(tensor_info.checked_expected_data_size()?).map_err(|_| {
                GGUFError::InvalidTensorData("Tensor size does not fit this platform".to_string())
            })?;
        if tensor_data.len() != expected_size {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor data size mismatch: expected {}, got {}",
                expected_size,
                tensor_data.len()
            )));
        }

        if self.tensor_name_index.contains(tensor_info.name()) {
            return Err(GGUFError::InvalidTensorData(format!(
                "Duplicate tensor name: '{}'",
                tensor_info.name()
            )));
        }

        let indexed_name = try_clone_tensor_name(tensor_info.name())?;
        self.tensors.try_reserve(1).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor descriptor".to_string())
        })?;
        self.tensor_name_index.try_reserve(1).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor-name index".to_string())
        })?;
        if !self.tensor_name_index.insert(indexed_name) {
            return Err(GGUFError::InvalidTensorData(format!(
                "Duplicate tensor name: '{}'",
                tensor_info.name()
            )));
        }
        self.tensors.push((tensor_info, tensor_data));
        if tensor_type.is_quantized() && self.metadata.get("general.quantization_version").is_none()
        {
            self.metadata.insert(
                "general.quantization_version".to_string(),
                crate::format::metadata::MetadataValue::U32(GGUF_QUANTIZATION_VERSION),
            );
        }
        Ok(())
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
        self.try_push_tensor(name.into(), shape, tensor_type, data)?;
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
        self.tensors
            .iter()
            .fold(0u64, |total, (info, _)| total.saturating_add(info.expected_data_size()))
    }

    /// Get tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|(info, _)| info.name()).collect()
    }

    /// Check if a tensor exists
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_name_index.contains(name)
    }

    /// Remove a tensor by name
    pub fn remove_tensor(mut self, name: &str) -> Self {
        self.tensors.retain(|(info, _)| info.name() != name);
        self.tensor_name_index.remove(name);
        self
    }

    /// Clear all tensors
    pub fn clear_tensors(mut self) -> Self {
        self.tensors.clear();
        self.tensor_name_index.clear();
        self
    }

    /// Clear all metadata
    pub fn clear_metadata(mut self) -> Self {
        self.metadata = Metadata::new();
        self
    }

    /// Build and write to a writer
    pub fn build_to_writer<W: Write>(self, writer: W) -> Result<GGUFWriteResult> {
        // Validate before building
        self.validate()?;

        let config = self.config.unwrap_or_default();
        let mut gguf_writer = GGUFFileWriter::with_config(writer, config);

        gguf_writer.write_complete_file(&self.metadata, &self.tensors)
    }

    /// Build and write to a file path
    pub fn build_to_file<P: AsRef<Path>>(self, path: P) -> Result<GGUFWriteResult> {
        self.validate()?;
        let config = self.config.clone().unwrap_or_default();
        create_gguf_file_with_config(path, &self.metadata, &self.tensors, config)
    }

    /// Build and return as bytes
    pub fn build_to_bytes(self) -> Result<(Vec<u8>, GGUFWriteResult)> {
        let mut buffer = Vec::new();
        let result = self.build_to_writer(&mut buffer)?;
        Ok((buffer, result))
    }

    /// Validate the builder state before building
    pub fn validate(&self) -> Result<()> {
        self.metadata.validate()?;
        if self.tensor_name_index.len() != self.tensors.len() {
            return Err(GGUFError::InvalidTensorData(
                "Tensor-name index is inconsistent with tensor descriptors".to_string(),
            ));
        }
        for (tensor_info, _) in &self.tensors {
            if !self.tensor_name_index.contains(tensor_info.name()) {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Tensor-name index is missing '{}'",
                    tensor_info.name()
                )));
            }
        }

        // Validate each tensor
        for (tensor_info, tensor_data) in &self.tensors {
            tensor_info.validate()?;
            tensor_data.validate()?;
        }

        if self.tensors.iter().any(|(info, _)| info.tensor_type().is_quantized()) {
            match self.metadata.get("general.quantization_version") {
                Some(crate::format::metadata::MetadataValue::U32(GGUF_QUANTIZATION_VERSION)) => {}
                Some(crate::format::metadata::MetadataValue::U32(version)) => {
                    return Err(GGUFError::InvalidMetadata(format!(
                        "general.quantization_version must be {}, got {}",
                        GGUF_QUANTIZATION_VERSION, version
                    )));
                }
                Some(value) => {
                    return Err(GGUFError::InvalidMetadata(format!(
                        "general.quantization_version must be u32, got {}",
                        value.value_type()
                    )));
                }
                None => {
                    return Err(GGUFError::InvalidMetadata(
                        "Quantized tensors require general.quantization_version".to_string(),
                    ));
                }
            }
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
    }

    /// Create a GGUF builder for a language model
    pub fn language_model<N>(name: N, context_length: u32, embedding_size: u32) -> Self
    where
        N: Into<String>,
    {
        use crate::format::metadata::MetadataValue;

        Self::simple(name, "Language Model")
            .add_metadata("llama.context_length", MetadataValue::U32(context_length))
            .add_metadata("llama.embedding_length", MetadataValue::U32(embedding_size))
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
            vec![embedding_size, vocab_size],
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

    /// Add a tensor with F32 data
    pub fn add_f32_tensor<N: Into<String>>(
        self,
        name: N,
        shape: Vec<u64>,
        data: Vec<f32>,
    ) -> Result<Self> {
        let byte_length = data.len().checked_mul(core::mem::size_of::<f32>()).ok_or_else(|| {
            GGUFError::InvalidTensorData("F32 tensor byte length overflows usize".to_string())
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(byte_length).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate F32 tensor byte buffer".to_string())
        })?;
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        self.add_tensor(name, shape, TensorType::F32, bytes)
    }

    /// Add a tensor with I32 data
    pub fn add_i32_tensor<N: Into<String>>(
        self,
        name: N,
        shape: Vec<u64>,
        data: Vec<i32>,
    ) -> Result<Self> {
        let byte_length = data.len().checked_mul(core::mem::size_of::<i32>()).ok_or_else(|| {
            GGUFError::InvalidTensorData("I32 tensor byte length overflows usize".to_string())
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(byte_length).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate I32 tensor byte buffer".to_string())
        })?;
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        self.add_tensor(name, shape, TensorType::I32, bytes)
    }

    /// Add a quantized tensor with raw quantized data
    pub fn add_quantized_tensor<N: Into<String>>(
        self,
        name: N,
        shape: Vec<u64>,
        tensor_type: TensorType,
        data: Vec<u8>,
    ) -> Result<Self> {
        if !tensor_type.is_quantized() {
            return Err(GGUFError::InvalidTensorData(format!(
                "add_quantized_tensor requires a quantized tensor type, got {}",
                tensor_type
            )));
        }
        self.add_tensor(name, shape, tensor_type, data)
    }
}

impl std::fmt::Display for GGUFBuilderSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GGUF Builder Summary:")?;
        writeln!(f, "  Tensors: {}", self.tensor_count)?;
        writeln!(f, "  Metadata entries: {}", self.metadata_count)?;
        writeln!(f, "  Total tensor size: {} bytes", self.total_tensor_size)?;
        writeln!(f, "  Tensor types:")?;
        let mut tensor_types: Vec<_> = self.tensor_types.iter().collect();
        tensor_types.sort_unstable_by(|(left, _), (right, _)| {
            (**left as u32)
                .cmp(&(**right as u32))
                .then_with(|| left.name().cmp(right.name()))
        });
        for (tensor_type, count) in tensor_types {
            writeln!(f, "    {}: {}", tensor_type.name(), count)?;
        }
        writeln!(f, "  Tensor names: {:?}", self.tensor_names)?;
        Ok(())
    }
}

fn try_clone_tensor_name(name: &str) -> Result<String> {
    let mut owned = String::new();
    owned
        .try_reserve_exact(name.len())
        .map_err(|_| GGUFError::InvalidTensorData("Unable to allocate tensor name".to_string()))?;
    owned.push_str(name);
    Ok(owned)
}

#[cfg(all(test, feature = "std"))]
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

        let data = vec![0u8; 4];
        builder
            .try_push_tensor(
                "dup".to_string(),
                vec![1],
                TensorType::F32,
                TensorData::new_owned(data.clone()),
            )
            .unwrap();
        let duplicate = builder.try_push_tensor(
            "dup".to_string(),
            vec![1],
            TensorType::F32,
            TensorData::new_owned(data),
        );

        assert!(matches!(
            duplicate,
            Err(GGUFError::InvalidTensorData(message))
                if message.contains("Duplicate tensor name")
        ));
        assert_eq!(builder.tensor_count(), 1);
        assert_eq!(builder.tensor_name_index.len(), 1);
        assert!(builder.validate().is_ok());
    }

    #[test]
    fn test_simple_builder() {
        let builder = GGUFBuilder::simple("test_model", "A test model");
        assert_eq!(builder.metadata_count(), 2);
        assert!(!builder.metadata.contains_key("general.file_type"));
    }

    #[test]
    fn test_language_model_builder() {
        let builder = GGUFBuilder::language_model("llama_test", 2048, 4096);
        assert!(builder.metadata_count() > 0);
        assert_eq!(builder.metadata.get("llama.context_length"), Some(&MetadataValue::U32(2048)));
        assert_eq!(builder.metadata.get("llama.embedding_length"), Some(&MetadataValue::U32(4096)));

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
        assert_eq!(builder.tensor_name_index.len(), 1);
        assert!(!builder.has_tensor("tensor1"));
        assert!(builder.has_tensor("tensor2"));

        let builder =
            builder.add_tensor("tensor1", vec![2], TensorType::F32, vec![0u8; 8]).unwrap();
        assert_eq!(builder.tensor_count(), 2);
        assert_eq!(builder.tensor_name_index.len(), 2);

        let builder = builder.clear_tensors();
        assert_eq!(builder.tensor_count(), 0);
        assert!(builder.tensor_name_index.is_empty());
        assert!(!builder.has_tensor("tensor1"));
        assert!(!builder.has_tensor("tensor2"));
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
        assert!(display.find("    F32:").unwrap() < display.find("    F16:").unwrap());
    }

    #[test]
    fn test_convenience_methods_return_errors_without_panicking() {
        assert!(GGUFBuilder::new()
            .add_f32_tensor("rank_five", vec![1, 1, 1, 1, 1], vec![0.0])
            .is_err());
        assert!(GGUFBuilder::new().add_i32_tensor("wrong_size", vec![2], vec![1]).is_err());
        assert!(GGUFBuilder::new()
            .add_quantized_tensor("invalid_block", vec![31], TensorType::Q4_0, vec![0; 18],)
            .is_err());
        assert!(GGUFBuilder::new()
            .add_quantized_tensor("not_quantized", vec![1], TensorType::F32, vec![0; 4],)
            .is_err());
    }

    #[test]
    fn test_quantized_tensors_declare_and_validate_quantization_version() {
        let builder = GGUFBuilder::new()
            .add_quantized_tensor("q4", vec![32], TensorType::Q4_0, vec![0; 18])
            .unwrap();
        assert_eq!(
            builder.metadata.get("general.quantization_version"),
            Some(&MetadataValue::U32(GGUF_QUANTIZATION_VERSION))
        );
        assert!(builder.validate().is_ok());

        let invalid_version = builder.add_metadata(
            "general.quantization_version",
            MetadataValue::U32(GGUF_QUANTIZATION_VERSION + 1),
        );
        assert!(invalid_version.validate().is_err());
    }

    #[test]
    fn test_vocabulary_uses_ggml_dimension_order() {
        let builder = GGUFBuilder::new().add_vocabulary(3, 2, vec![0; 3 * 2 * 4]).unwrap();
        assert_eq!(builder.tensors[0].0.shape().dims(), &[2, 3]);
    }
}
