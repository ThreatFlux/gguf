//! Tensor information and metadata structures

use crate::error::{GGUFError, Result};
use crate::format::constants::{GGUF_MAX_DIMENSIONS, GGUF_MAX_TENSOR_NAME_LENGTH};
use crate::format::types::GGUFTensorType as TensorType;
use crate::tensor::{TensorData, TensorShape};

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use hashbrown::HashMap;
#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

// Import core modules for no_std compatibility
#[cfg(not(feature = "std"))]
use core::fmt;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use libm::sqrt;

// Helper for the single-pass F32 statistics implementation.
#[cfg(feature = "std")]
fn sqrt_f64(value: f64) -> f64 {
    value.sqrt()
}

#[cfg(all(not(feature = "std"), feature = "alloc"))]
fn sqrt_f64(value: f64) -> f64 {
    sqrt(value)
}

#[cfg(not(any(feature = "std", feature = "alloc")))]
fn sqrt_f64(value: f64) -> f64 {
    // Newton-Raphson method for square root
    if value == 0.0 {
        return 0.0;
    }
    let mut guess = value / 2.0;
    for _ in 0..10 {
        guess = (guess + value / guess) / 2.0;
    }
    guess
}

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
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
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
    /// Non-cryptographic data checksum for caller-side comparison
    pub checksum: Option<u32>,
}

/// Tensor layout information
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLayout {
    /// Memory layout convention used by the stride vector
    pub memory_layout: MemoryLayout,
    /// Byte strides in descriptor dimension order.
    ///
    /// For quantized tensors, stride 0 is the byte size of one quantization
    /// block rather than the size of an individually addressable weight.
    pub strides: Vec<u64>,
    /// Whether the tensor is contiguous in memory
    pub is_contiguous: bool,
    /// Intrinsic alignment of the encoded byte representation.
    ///
    /// This is always 1 because this type describes serialized bytes, not a
    /// host ABI type. GGUF tensor-offset alignment is controlled separately by
    /// the file's `general.alignment` metadata and is enforced by readers and
    /// writers.
    pub alignment: usize,
}

/// Memory layout types
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLayout {
    /// GGML/GGUF layout: descriptor dimension 0 is contiguous
    Ggml,
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
    ///
    /// Returns `u64::MAX` if the shape/type combination cannot be represented. Parsing and
    /// writing code should use [`Self::checked_expected_data_size`] so malformed input is rejected.
    pub fn expected_data_size(&self) -> u64 {
        self.checked_expected_data_size().unwrap_or(u64::MAX)
    }

    /// Calculate the expected tensor size, rejecting invalid block layouts and overflow.
    pub fn checked_expected_data_size(&self) -> Result<u64> {
        let element_count = self.shape.checked_element_count().ok_or_else(|| {
            GGUFError::InvalidTensorData(format!(
                "Tensor '{}' element count overflows u64",
                self.name
            ))
        })?;

        if self.tensor_type.is_quantized() {
            let block_size = self.tensor_type.block_size();
            let first_dimension = self.shape.dims().first().copied().unwrap_or(0);
            if block_size == 0 {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' uses unsupported tensor type {}",
                    self.name,
                    self.tensor_type.name()
                )));
            }
            if first_dimension % block_size as u64 != 0 {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' first dimension {} is not divisible by {} for {}",
                    self.name,
                    first_dimension,
                    block_size,
                    self.tensor_type.name()
                )));
            }
        }

        self.tensor_type.checked_calculate_size(element_count).ok_or_else(|| {
            GGUFError::InvalidTensorData(format!(
                "Tensor '{}' data size overflows u64 or uses an unsupported type",
                self.name
            ))
        })
    }

    /// Validate that the tensor info is consistent
    pub fn validate(&self) -> Result<()> {
        // Validate name
        if self.name.is_empty() {
            return Err(GGUFError::InvalidTensorData("Tensor name cannot be empty".to_string()));
        }

        if self.name.len() > GGUF_MAX_TENSOR_NAME_LENGTH {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' name is {} bytes; maximum is {}",
                self.name,
                self.name.len(),
                GGUF_MAX_TENSOR_NAME_LENGTH
            )));
        }

        // Validate shape
        if self.shape.dimensions.is_empty() {
            return Err(GGUFError::InvalidTensorData("Tensor shape cannot be empty".to_string()));
        }

        if self.shape.ndim() > GGUF_MAX_DIMENSIONS {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' has too many dimensions: {}",
                self.name,
                self.shape.ndim()
            )));
        }

        let expected_size_u64 = self.checked_expected_data_size()?;

        // Validate data size if data is present
        if let Some(data) = &self.data {
            data.validate()?;
            let expected_size = usize::try_from(expected_size_u64).map_err(|_| {
                GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' data size does not fit this platform",
                    self.name
                ))
            })?;
            let actual_size = data.len();

            if actual_size != expected_size {
                #[cfg(any(feature = "std", feature = "alloc"))]
                return Err(GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' data size mismatch: expected {} bytes, got {} bytes",
                    self.name, expected_size, actual_size
                )));
                #[cfg(not(any(feature = "std", feature = "alloc")))]
                return Err(GGUFError::AllocationRequired);
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

    /// Calculate tensor layout information using canonical GGML block geometry.
    ///
    /// GGUF dimensions follow GGML ordering, where dimension 0 is contiguous.
    /// For scalar types, `nb[0]` is the scalar byte size. For quantized types,
    /// `nb[0]` is the encoded block byte size and `nb[1]` advances over all
    /// blocks in dimension 0.
    pub fn calculate_layout(&self) -> Result<TensorLayout> {
        self.validate()?;

        let block_size = u64::try_from(self.tensor_type.block_size()).map_err(|_| {
            GGUFError::InvalidTensorData(format!(
                "Tensor '{}' block size does not fit u64",
                self.name
            ))
        })?;
        let block_bytes = u64::try_from(self.tensor_type.block_size_bytes().ok_or_else(|| {
            GGUFError::InvalidTensorData(format!(
                "Tensor '{}' uses unsupported tensor type {}",
                self.name,
                self.tensor_type.name()
            ))
        })?)
        .map_err(|_| {
            GGUFError::InvalidTensorData(format!(
                "Tensor '{}' block byte size does not fit u64",
                self.name
            ))
        })?;

        let dimensions = self.shape.dims();
        let mut byte_strides = Vec::with_capacity(dimensions.len());
        byte_strides.push(block_bytes);

        if dimensions.len() > 1 {
            let blocks_in_dim0 = dimensions[0] / block_size;
            let mut stride = block_bytes.checked_mul(blocks_in_dim0).ok_or_else(|| {
                GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' byte stride overflows u64",
                    self.name
                ))
            })?;
            byte_strides.push(stride);

            for &dimension in &dimensions[1..dimensions.len() - 1] {
                stride = stride.checked_mul(dimension).ok_or_else(|| {
                    GGUFError::InvalidTensorData(format!(
                        "Tensor '{}' byte stride overflows u64",
                        self.name
                    ))
                })?;
                byte_strides.push(stride);
            }
        }

        Ok(TensorLayout {
            memory_layout: MemoryLayout::Ggml,
            strides: byte_strides,
            is_contiguous: self.shape.is_contiguous(),
            alignment: 1,
        })
    }

    /// Calculate decoded F32 statistics, or checksum-only information for
    /// encoded types that this crate does not decode.
    pub fn calculate_stats(&self) -> Option<TensorStats> {
        let data = self.data.as_ref()?;

        match self.tensor_type {
            TensorType::F32 => self.calculate_f32_stats(data),
            _ => Some(TensorStats {
                min_value: None,
                max_value: None,
                mean_value: None,
                std_deviation: None,
                zero_count: None,
                checksum: Some(data.checksum()),
            }),
        }
    }

    /// Calculate statistics for F32 data
    fn calculate_f32_stats(&self, data: &TensorData) -> Option<TensorStats> {
        let bytes = data.as_slice();
        if !bytes.len().is_multiple_of(4) {
            return None;
        }

        if bytes.is_empty() {
            return None;
        }

        let mut count = 0u64;
        let mut zero_count = 0u64;
        let mut min_value = f64::INFINITY;
        let mut max_value = f64::NEG_INFINITY;
        let mut mean = 0.0f64;
        let mut squared_deviation_sum = 0.0f64;

        for chunk in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes(chunk.try_into().ok()?) as f64;
            count += 1;
            if value == 0.0 {
                zero_count += 1;
            }
            min_value = min_value.min(value);
            max_value = max_value.max(value);

            // Welford's algorithm avoids both a temporary value buffer and
            // the cancellation error of sum-of-squares variance formulas.
            let delta = value - mean;
            mean += delta / count as f64;
            let delta_after_mean = value - mean;
            squared_deviation_sum += delta * delta_after_mean;
        }

        let variance = squared_deviation_sum / count as f64;

        Some(TensorStats {
            min_value: Some(min_value),
            max_value: Some(max_value),
            mean_value: Some(mean),
            std_deviation: Some(sqrt_f64(variance)),
            zero_count: Some(zero_count),
            checksum: Some(data.checksum()),
        })
    }

    /// Get a human-readable summary
    #[cfg(any(feature = "std", feature = "alloc"))]
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
        let expected_size = self
            .checked_expected_data_size()
            .ok()
            .and_then(|size| usize::try_from(size).ok())
            .unwrap_or(usize::MAX);
        let loaded_size = self.data.as_ref().map_or(0, |d| d.len());

        TensorMemoryInfo {
            name: self.name.clone(),
            expected_bytes: expected_size,
            loaded_bytes: loaded_size,
            is_loaded: self.has_data(),
            loaded_fraction: if expected_size > 0 {
                loaded_size as f32 / expected_size as f32
            } else {
                0.0
            },
        }
    }
}

/// Memory usage information for a tensor
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct TensorMemoryInfo {
    /// Tensor name
    pub name: String,
    /// Expected size in bytes
    pub expected_bytes: usize,
    /// Actually loaded bytes
    pub loaded_bytes: usize,
    /// Whether data is loaded
    pub is_loaded: bool,
    /// Fraction of the expected payload currently loaded (`loaded / expected`)
    pub loaded_fraction: f32,
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

#[cfg(feature = "std")]
impl std::fmt::Display for TensorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

#[cfg(not(feature = "std"))]
impl fmt::Display for TensorInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for MemoryLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryLayout::Ggml => write!(f, "GGML (dimension 0 contiguous)"),
            MemoryLayout::RowMajor => write!(f, "Row-Major"),
            MemoryLayout::ColumnMajor => write!(f, "Column-Major"),
            MemoryLayout::Custom => write!(f, "Custom"),
        }
    }
}

#[cfg(not(feature = "std"))]
impl fmt::Display for MemoryLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryLayout::Ggml => write!(f, "GgmlDim0Contiguous"),
            MemoryLayout::RowMajor => write!(f, "RowMajor"),
            MemoryLayout::ColumnMajor => write!(f, "ColumnMajor"),
            MemoryLayout::Custom => write!(f, "Custom"),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for TensorLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layout({}, strides: {:?}, contiguous: {}, align: {})",
            self.memory_layout, self.strides, self.is_contiguous, self.alignment
        )
    }
}

#[cfg(not(feature = "std"))]
impl fmt::Display for TensorLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorLayout {{ memory: {}, contiguous: {}, alignment: {} }}",
            self.memory_layout, self.is_contiguous, self.alignment
        )
    }
}

#[cfg(feature = "std")]
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

#[cfg(not(feature = "std"))]
impl fmt::Display for TensorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorStats {{ min: {:?}, max: {:?}, mean: {:?} }}",
            self.min_value, self.max_value, self.mean_value
        )
    }
}

#[cfg(all(test, feature = "std"))]
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

    #[cfg(feature = "mmap")]
    #[test]
    fn test_invalid_mapped_data_is_rejected() {
        use memmap2::Mmap;
        use std::{fs::File, io::Write, sync::Arc};
        use tempfile::NamedTempFile;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[0u8; 4]).unwrap();
        file.flush().unwrap();
        let mapped_file = File::open(file.path()).unwrap();
        let mmap = Arc::new(unsafe { Mmap::map(&mapped_file).unwrap() });
        let data = TensorData::Mapped { mmap, offset: usize::MAX, length: 4 };
        let info = TensorInfo::with_data(
            "mapped".to_string(),
            TensorShape::new(vec![1]).unwrap(),
            TensorType::F32,
            0,
            data,
        );

        assert!(info.validate().is_err());
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

        let layout = tensor_info.calculate_layout().unwrap();
        assert_eq!(layout.memory_layout, MemoryLayout::Ggml);
        assert_eq!(layout.strides, vec![4, 8]);
        assert!(layout.is_contiguous);
        assert_eq!(layout.alignment, 1);
    }

    #[test]
    fn test_scalar_tensor_layout() {
        let tensor_info =
            TensorInfo::new("scalar".to_string(), TensorShape::scalar(), TensorType::F32, 0);

        let layout = tensor_info.calculate_layout().unwrap();
        assert_eq!(layout.memory_layout, MemoryLayout::Ggml);
        assert_eq!(layout.strides, vec![4]);
    }

    #[test]
    fn test_quantized_tensor_layout_matches_ggml_nb_vector() {
        let q4_0 = TensorInfo::new(
            "q4_0".to_string(),
            TensorShape::new(vec![64, 3]).unwrap(),
            TensorType::Q4_0,
            0,
        );
        assert_eq!(q4_0.calculate_layout().unwrap().strides, vec![18, 36]);

        let q4_k = TensorInfo::new(
            "q4_k".to_string(),
            TensorShape::new(vec![256, 2, 3]).unwrap(),
            TensorType::Q4_K,
            0,
        );
        assert_eq!(q4_k.calculate_layout().unwrap().strides, vec![144, 144, 288]);
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
        assert_eq!(stats.min_value, Some(1.0));
        assert_eq!(stats.max_value, Some(2.0));
        assert_eq!(stats.mean_value, Some(1.5));
        assert_eq!(stats.std_deviation, Some(0.5));
        assert_eq!(stats.zero_count, Some(0));
        assert!(stats.checksum.is_some());
    }

    #[test]
    fn test_encoded_tensor_stats_do_not_count_zero_bytes_as_values() {
        let tensor_info = TensorInfo::with_data(
            "f16".to_string(),
            TensorShape::new(vec![2]).unwrap(),
            TensorType::F16,
            0,
            TensorData::new_owned(vec![0; 4]),
        );

        let stats = tensor_info.calculate_stats().unwrap();
        assert_eq!(stats.min_value, None);
        assert_eq!(stats.max_value, None);
        assert_eq!(stats.mean_value, None);
        assert_eq!(stats.std_deviation, None);
        assert_eq!(stats.zero_count, None);
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
        assert_eq!(memory_info.loaded_fraction, 1.0);
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

        let layout = tensor_info.calculate_layout().unwrap();
        let layout_str = format!("{}", layout);
        assert!(layout_str.contains("GGML"));

        let memory_layout_str = format!("{}", MemoryLayout::ColumnMajor);
        assert_eq!(memory_layout_str, "Column-Major");
    }
}
