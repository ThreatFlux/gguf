//! Tensor information and metadata structures

use crate::error::{GGUFError, Result};
use crate::format::types::GGUFTensorType as TensorType;
use crate::tensor::{TensorData, TensorShape};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete information about a tensor
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// Name of the tensor
    pub name: String,
    /// Shape of the tensor
    pub shape: TensorShape,
    /// Data type of the tensor
    pub tensor_type: TensorType,
    /// Offset in the GGUF file where data begins
    pub data_offset: u64,
    /// Actual tensor data (may be empty if not loaded)
    pub data: Option<TensorData>,
    /// Additional metadata
    pub metadata: TensorMetadata,
}

/// Metadata associated with a tensor
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TensorMetadata {
    /// Custom key-value pairs
    pub attributes: HashMap<String, String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Description of the tensor's purpose
    pub description: Option<String>,
    /// Source information (where this tensor came from)
    pub source: Option<String>,
    /// Version information
    pub version: Option<String>,
}

/// Summary statistics about a tensor
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorStats {
    /// Minimum value (for numerical analysis)
    pub min_value: Option<f64>,
    /// Maximum value
    pub max_value: Option<f64>,
    /// Mean value
    pub mean_value: Option<f64>,
    /// Standard deviation
    pub std_deviation: Option<f64>,
    /// Number of zero elements
    pub zero_count: Option<u64>,
    /// Data checksum for integrity checking
    pub checksum: Option<u32>,
}

/// Tensor layout information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorLayout {
    /// Memory layout (row-major, column-major, etc.)
    pub memory_layout: MemoryLayout,
    /// Stride information
    pub strides: Vec<u64>,
    /// Whether the tensor is contiguous in memory
    pub is_contiguous: bool,
    /// Alignment requirements
    pub alignment: usize,
}

/// Memory layout types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryLayout {
    /// Row-major (C-style) layout
    RowMajor,
    /// Column-major (Fortran-style) layout
    ColumnMajor,
    /// Custom layout
    Custom,
}

impl TensorInfo {
    /// Create a new tensor info
    pub fn new(
        name: String,
        shape: TensorShape,
        tensor_type: TensorType,
        data_offset: u64,
    ) -> Self {
        Self {
            name,
            shape,
            tensor_type,
            data_offset,
            data: None,
            metadata: TensorMetadata::default(),
        }
    }

    /// Create tensor info with data
    pub fn with_data(
        name: String,
        shape: TensorShape,
        tensor_type: TensorType,
        data_offset: u64,
        data: TensorData,
    ) -> Self {
        Self {
            name,
            shape,
            tensor_type,
            data_offset,
            data: Some(data),
            metadata: TensorMetadata::default(),
        }
    }

    /// Get the tensor name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the tensor shape
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Get the tensor type
    pub fn tensor_type(&self) -> TensorType {
        self.tensor_type
    }

    /// Get the data offset
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Get the tensor data if available
    pub fn data(&self) -> Option<&TensorData> {
        self.data.as_ref()
    }

    /// Get mutable tensor data if available
    pub fn data_mut(&mut self) -> Option<&mut TensorData> {
        self.data.as_mut()
    }

    /// Take ownership of the tensor data
    pub fn take_data(&mut self) -> Option<TensorData> {
        self.data.take()
    }

    /// Set the tensor data
    pub fn set_data(&mut self, data: TensorData) {
        self.data = Some(data);
    }

    /// Clear the tensor data (useful for memory management)
    pub fn clear_data(&mut self) {
        self.data = None;
    }

    /// Check if tensor data is loaded
    pub fn has_data(&self) -> bool {
        self.data.is_some()
    }

    /// Calculate the number of elements in the tensor
    pub fn element_count(&self) -> u64 {
        self.shape.element_count()
    }

    /// Calculate the expected size of the tensor data in bytes
    pub fn expected_data_size(&self) -> u64 {
        self.tensor_type.calculate_size(self.element_count())
    }

    /// Validate that the tensor info is consistent
    pub fn validate(&self) -> Result<()> {
        // Validate name
        if self.name.is_empty() {
            return Err(GGUFError::InvalidTensorData("Tensor name cannot be empty".to_string()));
        }

        // Validate shape
        if self.shape.dimensions.is_empty() {
            return Err(GGUFError::InvalidTensorData("Tensor shape cannot be empty".to_string()));
        }

        // Validate data size if data is present
        if let Some(data) = &self.data {
            let expected_size = self.expected_data_size() as usize;
            let actual_size = data.len();

            if actual_size != expected_size {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' data size mismatch: expected {} bytes, got {} bytes",
                    self.name, expected_size, actual_size
                )));
            }
        }

        Ok(())
    }

    /// Get metadata reference
    pub fn metadata(&self) -> &TensorMetadata {
        &self.metadata
    }

    /// Get mutable metadata reference
    pub fn metadata_mut(&mut self) -> &mut TensorMetadata {
        &mut self.metadata
    }

    /// Calculate tensor layout information
    pub fn calculate_layout(&self) -> TensorLayout {
        let element_strides = self.shape.calculate_strides();
        let byte_strides = element_strides
            .iter()
            .map(|&s| s * self.tensor_type.element_size() as u64)
            .collect();

        TensorLayout {
            memory_layout: MemoryLayout::RowMajor, // Default to row-major
            strides: byte_strides,
            is_contiguous: self.shape.is_contiguous(),
            alignment: self.tensor_type.element_size(),
        }
    }

    /// Calculate basic statistics if data is available and numeric
    pub fn calculate_stats(&self) -> Option<TensorStats> {
        let data = self.data.as_ref()?;

        // Only calculate stats for floating point types
        match self.tensor_type {
            TensorType::F32 => self.calculate_f32_stats(data),
            TensorType::F16 | TensorType::BF16 => self.calculate_f16_stats(data),
            _ => Some(TensorStats {
                min_value: None,
                max_value: None,
                mean_value: None,
                std_deviation: None,
                zero_count: self.count_zero_bytes(data),
                checksum: Some(data.checksum()),
            }),
        }
    }

    /// Calculate statistics for F32 data
    fn calculate_f32_stats(&self, data: &TensorData) -> Option<TensorStats> {
        let bytes = data.as_slice();
        if bytes.len() % 4 != 0 {
            return None;
        }

        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        if values.is_empty() {
            return None;
        }

        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean = values.iter().sum::<f32>() / values.len() as f32;

        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();

        let zero_count = values.iter().filter(|&&x| x == 0.0).count() as u64;

        Some(TensorStats {
            min_value: Some(min_val as f64),
            max_value: Some(max_val as f64),
            mean_value: Some(mean as f64),
            std_deviation: Some(std_dev as f64),
            zero_count: Some(zero_count),
            checksum: Some(data.checksum()),
        })
    }

    /// Calculate statistics for F16/BF16 data (simplified)
    fn calculate_f16_stats(&self, data: &TensorData) -> Option<TensorStats> {
        // For now, just return basic info since F16 conversion is complex
        Some(TensorStats {
            min_value: None,
            max_value: None,
            mean_value: None,
            std_deviation: None,
            zero_count: self.count_zero_bytes(data),
            checksum: Some(data.checksum()),
        })
    }

    /// Count zero bytes in data
    fn count_zero_bytes(&self, data: &TensorData) -> Option<u64> {
        let zero_count = data.as_slice().iter().filter(|&&b| b == 0).count();
        Some(zero_count as u64)
    }

    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Tensor '{}': {} ({}), {} elements, offset: {}{}",
            self.name,
            self.shape,
            self.tensor_type.name(),
            self.element_count(),
            self.data_offset,
            if self.has_data() { ", data loaded" } else { "" }
        )
    }

    /// Check if this tensor is compatible with another for operations
    pub fn is_compatible_with(&self, other: &TensorInfo) -> bool {
        self.tensor_type == other.tensor_type && self.shape.is_elementwise_compatible(&other.shape)
    }

    /// Get memory usage information
    pub fn memory_usage(&self) -> TensorMemoryInfo {
        let expected_size = self.expected_data_size() as usize;
        let loaded_size = self.data.as_ref().map_or(0, |d| d.len());

        TensorMemoryInfo {
            name: self.name.clone(),
            expected_bytes: expected_size,
            loaded_bytes: loaded_size,
            is_loaded: self.has_data(),
            compression_ratio: if expected_size > 0 {
                loaded_size as f32 / expected_size as f32
            } else {
                0.0
            },
        }
    }
}

/// Memory usage information for a tensor
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorMemoryInfo {
    /// Tensor name
    pub name: String,
    /// Expected size in bytes
    pub expected_bytes: usize,
    /// Actually loaded bytes
    pub loaded_bytes: usize,
    /// Whether data is loaded
    pub is_loaded: bool,
    /// Compression ratio (actual/expected)
    pub compression_ratio: f32,
}

impl TensorMetadata {
    /// Create new empty metadata
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an attribute
    pub fn add_attribute<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.attributes.insert(key.into(), value.into());
    }

    /// Get an attribute value
    pub fn get_attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(|s| s.as_str())
    }

    /// Add a tag
    pub fn add_tag<T: Into<String>>(&mut self, tag: T) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Remove a tag
    pub fn remove_tag(&mut self, tag: &str) {
        self.tags.retain(|t| t != tag);
    }

    /// Check if has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(&tag.to_string())
    }

    /// Set description
    pub fn set_description<S: Into<String>>(&mut self, description: S) {
        self.description = Some(description.into());
    }

    /// Set source
    pub fn set_source<S: Into<String>>(&mut self, source: S) {
        self.source = Some(source.into());
    }

    /// Set version
    pub fn set_version<S: Into<String>>(&mut self, version: S) {
        self.version = Some(version.into());
    }

    /// Check if metadata is empty
    pub fn is_empty(&self) -> bool {
        self.attributes.is_empty()
            && self.tags.is_empty()
            && self.description.is_none()
            && self.source.is_none()
            && self.version.is_none()
    }
}

impl std::fmt::Display for TensorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

impl std::fmt::Display for MemoryLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryLayout::RowMajor => write!(f, "Row-Major"),
            MemoryLayout::ColumnMajor => write!(f, "Column-Major"),
            MemoryLayout::Custom => write!(f, "Custom"),
        }
    }
}

impl std::fmt::Display for TensorLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layout({}, strides: {:?}, contiguous: {}, align: {})",
            self.memory_layout, self.strides, self.is_contiguous, self.alignment
        )
    }
}

impl std::fmt::Display for TensorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stats(")?;

        let mut parts = Vec::new();

        if let (Some(min), Some(max)) = (self.min_value, self.max_value) {
            parts.push(format!("range: [{:.4}, {:.4}]", min, max));
        }

        if let Some(mean) = self.mean_value {
            parts.push(format!("mean: {:.4}", mean));
        }

        if let Some(std) = self.std_deviation {
            parts.push(format!("std: {:.4}", std));
        }

        if let Some(zeros) = self.zero_count {
            parts.push(format!("zeros: {}", zeros));
        }

        write!(f, "{})", parts.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorShape;

    #[test]
    fn test_tensor_info_creation() {
        let shape = TensorShape::new(vec![2, 3]).unwrap();
        let tensor_info =
            TensorInfo::new("test_tensor".to_string(), shape.clone(), TensorType::F32, 1024);

        assert_eq!(tensor_info.name(), "test_tensor");
        assert_eq!(*tensor_info.shape(), shape);
        assert_eq!(tensor_info.tensor_type(), TensorType::F32);
        assert_eq!(tensor_info.data_offset(), 1024);
        assert!(!tensor_info.has_data());
        assert_eq!(tensor_info.element_count(), 6);
        assert_eq!(tensor_info.expected_data_size(), 24); // 6 elements * 4 bytes
    }

    #[test]
    fn test_tensor_info_with_data() {
        let shape = TensorShape::new(vec![2, 2]).unwrap();
        let data = TensorData::new_owned(vec![0u8; 16]); // 4 F32 values = 16 bytes
        let tensor_info =
            TensorInfo::with_data("data_tensor".to_string(), shape, TensorType::F32, 0, data);

        assert!(tensor_info.has_data());
        assert!(tensor_info.validate().is_ok());
    }

    #[test]
    fn test_tensor_info_validation() {
        let shape = TensorShape::new(vec![2]).unwrap();
        let mut tensor_info = TensorInfo::new("test".to_string(), shape, TensorType::F32, 0);

        // Valid case
        assert!(tensor_info.validate().is_ok());

        // Invalid: empty name
        tensor_info.name = String::new();
        assert!(tensor_info.validate().is_err());

        tensor_info.name = "test".to_string();

        // Invalid: wrong data size
        let wrong_data = TensorData::new_owned(vec![0u8; 3]); // Should be 8 bytes for 2 F32
        tensor_info.set_data(wrong_data);
        assert!(tensor_info.validate().is_err());
    }

    #[test]
    fn test_tensor_metadata() {
        let mut metadata = TensorMetadata::new();

        metadata.add_attribute("key1", "value1");
        metadata.add_tag("tag1");
        metadata.set_description("Test tensor");

        assert_eq!(metadata.get_attribute("key1"), Some("value1"));
        assert!(metadata.has_tag("tag1"));
        assert_eq!(metadata.description, Some("Test tensor".to_string()));
        assert!(!metadata.is_empty());

        // Test tag operations
        metadata.add_tag("tag2");
        assert!(metadata.has_tag("tag2"));

        metadata.remove_tag("tag1");
        assert!(!metadata.has_tag("tag1"));
        assert!(metadata.has_tag("tag2"));
    }

    #[test]
    fn test_tensor_layout() {
        let shape = TensorShape::new(vec![2, 3]).unwrap();
        let tensor_info = TensorInfo::new("test".to_string(), shape, TensorType::F32, 0);

        let layout = tensor_info.calculate_layout();
        assert_eq!(layout.memory_layout, MemoryLayout::RowMajor);
        assert_eq!(layout.strides, vec![12, 4]); // 3*4 bytes, 1*4 bytes
        assert!(layout.is_contiguous);
        assert_eq!(layout.alignment, 4); // F32 alignment
    }

    #[test]
    fn test_tensor_stats_f32() {
        let shape = TensorShape::new(vec![2]).unwrap();
        let data = TensorData::new_owned(vec![
            0x00, 0x00, 0x80, 0x3f, // 1.0 in little-endian F32
            0x00, 0x00, 0x00, 0x40, // 2.0 in little-endian F32
        ]);

        let tensor_info =
            TensorInfo::with_data("test".to_string(), shape, TensorType::F32, 0, data);

        let stats = tensor_info.calculate_stats();
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert!(stats.min_value.is_some());
        assert!(stats.max_value.is_some());
        assert!(stats.mean_value.is_some());
        assert!(stats.checksum.is_some());
    }

    #[test]
    fn test_tensor_compatibility() {
        let shape1 = TensorShape::new(vec![2, 3]).unwrap();
        let shape2 = TensorShape::new(vec![2, 3]).unwrap();
        let shape3 = TensorShape::new(vec![3, 2]).unwrap();

        let tensor1 = TensorInfo::new("t1".to_string(), shape1, TensorType::F32, 0);
        let tensor2 = TensorInfo::new("t2".to_string(), shape2.clone(), TensorType::F32, 0);
        let tensor3 = TensorInfo::new("t3".to_string(), shape3, TensorType::F32, 0);
        let tensor4 = TensorInfo::new("t4".to_string(), shape2.clone(), TensorType::F16, 0);

        assert!(tensor1.is_compatible_with(&tensor2)); // Same shape and type
        assert!(!tensor1.is_compatible_with(&tensor3)); // Different shape
        assert!(!tensor1.is_compatible_with(&tensor4)); // Different type
    }

    #[test]
    fn test_memory_usage_info() {
        let shape = TensorShape::new(vec![10]).unwrap();
        let data = TensorData::new_owned(vec![0u8; 40]); // 10 F32 values

        let tensor_info =
            TensorInfo::with_data("memory_test".to_string(), shape, TensorType::F32, 0, data);

        let memory_info = tensor_info.memory_usage();
        assert_eq!(memory_info.name, "memory_test");
        assert_eq!(memory_info.expected_bytes, 40);
        assert_eq!(memory_info.loaded_bytes, 40);
        assert!(memory_info.is_loaded);
        assert_eq!(memory_info.compression_ratio, 1.0);
    }

    #[test]
    fn test_tensor_info_summary() {
        let shape = TensorShape::new(vec![2, 3]).unwrap();
        let tensor_info =
            TensorInfo::new("summary_test".to_string(), shape, TensorType::Q4_0, 2048);

        let summary = tensor_info.summary();
        assert!(summary.contains("summary_test"));
        assert!(summary.contains("Q4_0"));
        assert!(summary.contains("6 elements"));
        assert!(summary.contains("2048"));
    }

    #[test]
    fn test_tensor_data_operations() {
        let shape = TensorShape::new(vec![2]).unwrap();
        let data = TensorData::new_owned(vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let mut tensor_info =
            TensorInfo::with_data("ops_test".to_string(), shape, TensorType::F32, 0, data);

        assert!(tensor_info.has_data());

        let taken_data = tensor_info.take_data();
        assert!(taken_data.is_some());
        assert!(!tensor_info.has_data());

        tensor_info.set_data(taken_data.unwrap());
        assert!(tensor_info.has_data());

        tensor_info.clear_data();
        assert!(!tensor_info.has_data());
    }

    #[test]
    fn test_display_impls() {
        let shape = TensorShape::new(vec![2, 3]).unwrap();
        let tensor_info = TensorInfo::new("display_test".to_string(), shape, TensorType::F32, 0);

        let display_str = format!("{}", tensor_info);
        assert!(!display_str.is_empty());

        let layout = tensor_info.calculate_layout();
        let layout_str = format!("{}", layout);
        assert!(layout_str.contains("Row-Major"));

        let memory_layout_str = format!("{}", MemoryLayout::ColumnMajor);
        assert_eq!(memory_layout_str, "Column-Major");
    }
}
