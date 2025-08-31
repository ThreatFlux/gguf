//! Tensor data storage and management

use crate::error::{GGUFError, Result};
#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "std")]
use std::sync::Arc;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{format, sync::Arc, vec, vec::Vec};

/// Container for tensor data with different storage backends
#[derive(Debug, Clone)]
pub enum TensorData {
    /// Owned byte vector (most common case)
    Owned(Vec<u8>),

    /// Reference to borrowed data (zero-copy when possible)
    Borrowed(&'static [u8]),

    /// Shared reference-counted data
    Shared(Arc<Vec<u8>>),

    /// Memory-mapped data (when using mmap feature)
    #[cfg(feature = "mmap")]
    Mapped {
        /// The memory map
        mmap: Arc<memmap2::Mmap>,
        /// Offset within the map
        offset: usize,
        /// Length of the data
        length: usize,
    },

    /// Empty/uninitialized data
    Empty,
}

/// Information about tensor data storage
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDataInfo {
    /// Size of the data in bytes
    pub size_bytes: usize,
    /// Storage type
    pub storage_type: TensorStorageType,
    /// Whether the data is aligned
    pub is_aligned: bool,
    /// Alignment boundary (if applicable)
    pub alignment: Option<usize>,
}

/// Types of tensor storage
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorStorageType {
    /// Owned in-memory storage
    Owned,
    /// Borrowed from external source
    Borrowed,
    /// Shared reference-counted storage
    Shared,
    /// Memory-mapped file storage
    Mapped,
    /// Empty/uninitialized
    Empty,
}

impl TensorData {
    /// Create new owned tensor data
    pub fn new_owned(data: Vec<u8>) -> Self {
        Self::Owned(data)
    }

    /// Create new borrowed tensor data
    pub fn new_borrowed(data: &'static [u8]) -> Self {
        Self::Borrowed(data)
    }

    /// Create new shared tensor data
    pub fn new_shared(data: Vec<u8>) -> Self {
        Self::Shared(Arc::new(data))
    }

    /// Create empty tensor data
    pub fn empty() -> Self {
        Self::Empty
    }

    /// Create tensor data with specified size filled with zeros
    pub fn zeros(size: usize) -> Self {
        Self::Owned(vec![0u8; size])
    }

    /// Create tensor data with specified size filled with a value
    pub fn filled(size: usize, value: u8) -> Self {
        Self::Owned(vec![value; size])
    }

    /// Create memory-mapped tensor data
    #[cfg(feature = "mmap")]
    pub fn new_mapped(mmap: Arc<memmap2::Mmap>, offset: usize, length: usize) -> Result<Self> {
        if offset + length > mmap.len() {
            return Err(GGUFError::InvalidTensorData(
                "Mapped region exceeds mmap bounds".to_string(),
            ));
        }

        Ok(Self::Mapped { mmap, offset, length })
    }

    /// Get the length of the tensor data in bytes
    pub fn len(&self) -> usize {
        match self {
            TensorData::Owned(data) => data.len(),
            TensorData::Borrowed(data) => data.len(),
            TensorData::Shared(data) => data.len(),
            #[cfg(feature = "mmap")]
            TensorData::Mapped { length, .. } => *length,
            TensorData::Empty => 0,
        }
    }

    /// Check if the tensor data is empty
    pub fn is_empty(&self) -> bool {
        match self {
            TensorData::Owned(data) => data.is_empty(),
            TensorData::Borrowed(data) => data.is_empty(),
            TensorData::Shared(data) => data.is_empty(),
            #[cfg(feature = "mmap")]
            TensorData::Mapped { length, .. } => *length == 0,
            TensorData::Empty => true,
        }
    }

    /// Get a slice of the tensor data
    pub fn as_slice(&self) -> &[u8] {
        match self {
            TensorData::Owned(data) => data,
            TensorData::Borrowed(data) => data,
            TensorData::Shared(data) => data,
            #[cfg(feature = "mmap")]
            TensorData::Mapped { mmap, offset, length } => &mmap[*offset..*offset + *length],
            TensorData::Empty => &[],
        }
    }

    /// Get mutable access to the data (only for owned data)
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        match self {
            TensorData::Owned(data) => Some(data),
            _ => None,
        }
    }

    /// Convert to owned data (cloning if necessary)
    pub fn to_owned(&self) -> Vec<u8> {
        self.as_slice().to_vec()
    }

    /// Take ownership of the data if possible, otherwise clone
    pub fn into_owned(self) -> Vec<u8> {
        match self {
            TensorData::Owned(data) => data,
            TensorData::Borrowed(data) => data.to_vec(),
            TensorData::Shared(data) => {
                // Try to unwrap if this is the only reference
                Arc::try_unwrap(data).unwrap_or_else(|shared| (*shared).clone())
            }
            #[cfg(feature = "mmap")]
            TensorData::Mapped { mmap, offset, length } => mmap[offset..offset + length].to_vec(),
            TensorData::Empty => Vec::new(),
        }
    }

    /// Get the storage type
    pub fn storage_type(&self) -> TensorStorageType {
        match self {
            TensorData::Owned(_) => TensorStorageType::Owned,
            TensorData::Borrowed(_) => TensorStorageType::Borrowed,
            TensorData::Shared(_) => TensorStorageType::Shared,
            #[cfg(feature = "mmap")]
            TensorData::Mapped { .. } => TensorStorageType::Mapped,
            TensorData::Empty => TensorStorageType::Empty,
        }
    }

    /// Get storage information
    pub fn storage_info(&self) -> TensorDataInfo {
        TensorDataInfo {
            size_bytes: self.len(),
            storage_type: self.storage_type(),
            is_aligned: self.is_aligned(),
            alignment: self.alignment(),
        }
    }

    /// Check if the data is properly aligned
    pub fn is_aligned(&self) -> bool {
        self.is_aligned_to(std::mem::align_of::<u64>())
    }

    /// Check if the data is aligned to a specific boundary
    pub fn is_aligned_to(&self, alignment: usize) -> bool {
        if alignment <= 1 {
            return true;
        }

        let ptr = self.as_slice().as_ptr() as usize;
        ptr % alignment == 0
    }

    /// Get the alignment of the data pointer
    pub fn alignment(&self) -> Option<usize> {
        let ptr = self.as_slice().as_ptr() as usize;

        if ptr == 0 {
            return None;
        }

        // Find the largest power of 2 that divides the address
        let mut alignment = 1;
        let mut test_ptr = ptr;

        while test_ptr % 2 == 0 && alignment < 4096 {
            alignment *= 2;
            test_ptr /= 2;
        }

        Some(alignment)
    }

    /// Create a slice of the tensor data
    pub fn slice(&self, start: usize, length: usize) -> Result<TensorData> {
        if start + length > self.len() {
            return Err(GGUFError::InvalidTensorData(
                "Slice bounds exceed data length".to_string(),
            ));
        }

        match self {
            TensorData::Owned(data) => Ok(TensorData::Owned(data[start..start + length].to_vec())),
            TensorData::Borrowed(data) => Ok(TensorData::Borrowed(&data[start..start + length])),
            TensorData::Shared(data) => Ok(TensorData::Owned(data[start..start + length].to_vec())),
            #[cfg(feature = "mmap")]
            TensorData::Mapped { mmap, offset, .. } => {
                TensorData::new_mapped(mmap.clone(), offset + start, length)
            }
            TensorData::Empty => {
                if start == 0 && length == 0 {
                    Ok(TensorData::Empty)
                } else {
                    Err(GGUFError::InvalidTensorData("Cannot slice empty tensor data".to_string()))
                }
            }
        }
    }

    /// Concatenate with another tensor data
    pub fn concat(&self, other: &TensorData) -> TensorData {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }

        let mut result = self.to_owned();
        result.extend_from_slice(other.as_slice());
        TensorData::Owned(result)
    }

    /// Check if two tensor data instances have the same content
    pub fn content_equals(&self, other: &TensorData) -> bool {
        self.as_slice() == other.as_slice()
    }

    /// Get a hexadecimal representation of the first few bytes (for debugging)
    pub fn hex_preview(&self, max_bytes: usize) -> String {
        let data = self.as_slice();
        let preview_len = max_bytes.min(data.len());

        if preview_len == 0 {
            return "[]".to_string();
        }

        let hex: String = data[..preview_len]
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(" ");

        if data.len() > max_bytes {
            format!("[{} ... ({} more bytes)]", hex, data.len() - preview_len)
        } else {
            format!("[{}]", hex)
        }
    }

    /// Validate the tensor data for basic consistency
    pub fn validate(&self) -> Result<()> {
        match self {
            TensorData::Empty => Ok(()),
            _ => {
                if self.as_slice().is_empty() {
                    return Err(GGUFError::InvalidTensorData(
                        "Non-empty tensor data has empty slice".to_string(),
                    ));
                }
                Ok(())
            }
        }
    }

    /// Calculate a simple checksum of the data (for integrity checking)
    pub fn checksum(&self) -> u32 {
        let data = self.as_slice();
        let mut checksum = 0u32;

        for (i, &byte) in data.iter().enumerate() {
            checksum = checksum.wrapping_add((byte as u32) << (i % 24));
            checksum = checksum.wrapping_mul(0x9e37_79b9); // Mixing constant
        }

        checksum
    }

    /// Get memory usage information
    pub fn memory_usage(&self) -> TensorMemoryUsage {
        match self {
            TensorData::Owned(data) => TensorMemoryUsage {
                allocated_bytes: data.capacity(),
                used_bytes: data.len(),
                is_shared: false,
                is_mapped: false,
            },
            TensorData::Borrowed(_) => TensorMemoryUsage {
                allocated_bytes: 0, // We don't own the memory
                used_bytes: self.len(),
                is_shared: false,
                is_mapped: false,
            },
            TensorData::Shared(data) => TensorMemoryUsage {
                allocated_bytes: data.capacity(),
                used_bytes: data.len(),
                is_shared: true,
                is_mapped: false,
            },
            #[cfg(feature = "mmap")]
            TensorData::Mapped { length, .. } => TensorMemoryUsage {
                allocated_bytes: 0, // Memory mapped
                used_bytes: *length,
                is_shared: true,
                is_mapped: true,
            },
            TensorData::Empty => TensorMemoryUsage {
                allocated_bytes: 0,
                used_bytes: 0,
                is_shared: false,
                is_mapped: false,
            },
        }
    }
}

/// Memory usage information for tensor data
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMemoryUsage {
    /// Allocated memory in bytes
    pub allocated_bytes: usize,
    /// Used memory in bytes
    pub used_bytes: usize,
    /// Whether the memory is shared
    pub is_shared: bool,
    /// Whether the memory is mapped from a file
    pub is_mapped: bool,
}

impl TensorMemoryUsage {
    /// Calculate the memory efficiency (used / allocated)
    pub fn efficiency(&self) -> f32 {
        if self.allocated_bytes == 0 {
            1.0 // Perfect efficiency for borrowed/mapped data
        } else {
            self.used_bytes as f32 / self.allocated_bytes as f32
        }
    }

    /// Get overhead in bytes
    pub fn overhead_bytes(&self) -> usize {
        self.allocated_bytes.saturating_sub(self.used_bytes)
    }
}

impl Default for TensorData {
    fn default() -> Self {
        Self::Empty
    }
}

impl PartialEq for TensorData {
    fn eq(&self, other: &Self) -> bool {
        self.content_equals(other)
    }
}

impl std::fmt::Display for TensorData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorData({:?}, {} bytes, {})",
            self.storage_type(),
            self.len(),
            self.hex_preview(8)
        )
    }
}

impl std::fmt::Display for TensorStorageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorStorageType::Owned => write!(f, "Owned"),
            TensorStorageType::Borrowed => write!(f, "Borrowed"),
            TensorStorageType::Shared => write!(f, "Shared"),
            TensorStorageType::Mapped => write!(f, "Mapped"),
            TensorStorageType::Empty => write!(f, "Empty"),
        }
    }
}

impl std::fmt::Display for TensorDataInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorDataInfo {{ type: {}, size: {} bytes, aligned: {} }}",
            self.storage_type, self.size_bytes, self.is_aligned
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_owned() {
        let data = vec![1, 2, 3, 4, 5];
        let tensor_data = TensorData::new_owned(data.clone());

        assert_eq!(tensor_data.len(), 5);
        assert!(!tensor_data.is_empty());
        assert_eq!(tensor_data.as_slice(), &data);
        assert_eq!(tensor_data.storage_type(), TensorStorageType::Owned);
    }

    #[test]
    fn test_tensor_data_empty() {
        let tensor_data = TensorData::empty();

        assert_eq!(tensor_data.len(), 0);
        assert!(tensor_data.is_empty());
        assert_eq!(tensor_data.as_slice(), &[] as &[u8]);
        assert_eq!(tensor_data.storage_type(), TensorStorageType::Empty);
    }

    #[test]
    fn test_tensor_data_zeros() {
        let tensor_data = TensorData::zeros(10);

        assert_eq!(tensor_data.len(), 10);
        assert_eq!(tensor_data.as_slice(), &vec![0u8; 10]);
    }

    #[test]
    fn test_tensor_data_filled() {
        let tensor_data = TensorData::filled(5, 42);

        assert_eq!(tensor_data.len(), 5);
        assert_eq!(tensor_data.as_slice(), &vec![42u8; 5]);
    }

    #[test]
    fn test_tensor_data_shared() {
        let data = vec![1, 2, 3, 4];
        let tensor_data = TensorData::new_shared(data.clone());

        assert_eq!(tensor_data.len(), 4);
        assert_eq!(tensor_data.as_slice(), &data);
        assert_eq!(tensor_data.storage_type(), TensorStorageType::Shared);
    }

    #[test]
    fn test_tensor_data_to_owned() {
        let data = vec![1, 2, 3, 4];
        let tensor_data = TensorData::new_shared(data.clone());
        let owned = tensor_data.to_owned();

        assert_eq!(owned, data);
    }

    #[test]
    fn test_tensor_data_into_owned() {
        let data = vec![1, 2, 3, 4];
        let tensor_data = TensorData::new_owned(data.clone());
        let owned = tensor_data.into_owned();

        assert_eq!(owned, data);
    }

    #[test]
    fn test_tensor_data_slice() {
        let data = vec![1, 2, 3, 4, 5];
        let tensor_data = TensorData::new_owned(data);

        let slice = tensor_data.slice(1, 3).unwrap();
        assert_eq!(slice.as_slice(), &[2, 3, 4]);

        // Test bounds checking
        assert!(tensor_data.slice(3, 5).is_err());
    }

    #[test]
    fn test_tensor_data_concat() {
        let data1 = TensorData::new_owned(vec![1, 2, 3]);
        let data2 = TensorData::new_owned(vec![4, 5, 6]);

        let concatenated = data1.concat(&data2);
        assert_eq!(concatenated.as_slice(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_tensor_data_content_equals() {
        let data1 = TensorData::new_owned(vec![1, 2, 3]);
        let data2 = TensorData::new_shared(vec![1, 2, 3]);
        let data3 = TensorData::new_owned(vec![1, 2, 4]);

        assert!(data1.content_equals(&data2));
        assert!(!data1.content_equals(&data3));
    }

    #[test]
    fn test_tensor_data_hex_preview() {
        let data = TensorData::new_owned(vec![0xde, 0xad, 0xbe, 0xef]);
        let preview = data.hex_preview(4);
        assert!(preview.contains("de"));
        assert!(preview.contains("ad"));
        assert!(preview.contains("be"));
        assert!(preview.contains("ef"));
    }

    #[test]
    fn test_tensor_data_checksum() {
        let data1 = TensorData::new_owned(vec![1, 2, 3, 4]);
        let data2 = TensorData::new_owned(vec![1, 2, 3, 4]);
        let data3 = TensorData::new_owned(vec![1, 2, 3, 5]);

        assert_eq!(data1.checksum(), data2.checksum());
        assert_ne!(data1.checksum(), data3.checksum());
    }

    #[test]
    fn test_tensor_data_validation() {
        let valid_data = TensorData::new_owned(vec![1, 2, 3]);
        assert!(valid_data.validate().is_ok());

        let empty_data = TensorData::empty();
        assert!(empty_data.validate().is_ok());
    }

    #[test]
    fn test_tensor_data_memory_usage() {
        let data = TensorData::new_owned(vec![1, 2, 3, 4, 5]);
        let usage = data.memory_usage();

        assert!(usage.used_bytes >= 5);
        assert!(usage.allocated_bytes >= usage.used_bytes);
        assert!(!usage.is_shared);
        assert!(!usage.is_mapped);
        assert!(usage.efficiency() > 0.0);
    }

    #[test]
    fn test_tensor_storage_info() {
        let data = TensorData::new_owned(vec![1, 2, 3, 4]);
        let info = data.storage_info();

        assert_eq!(info.size_bytes, 4);
        assert_eq!(info.storage_type, TensorStorageType::Owned);
    }

    #[test]
    fn test_memory_usage_efficiency() {
        let usage = TensorMemoryUsage {
            allocated_bytes: 100,
            used_bytes: 80,
            is_shared: false,
            is_mapped: false,
        };

        assert_eq!(usage.efficiency(), 0.8);
        assert_eq!(usage.overhead_bytes(), 20);
    }

    #[test]
    fn test_tensor_data_display() {
        let data = TensorData::new_owned(vec![1, 2, 3]);
        let display_str = format!("{}", data);

        assert!(display_str.contains("TensorData"));
        assert!(display_str.contains("Owned"));
        assert!(display_str.contains("3 bytes"));
    }

    #[test]
    fn test_tensor_data_equality() {
        let data1 = TensorData::new_owned(vec![1, 2, 3]);
        let data2 = TensorData::new_shared(vec![1, 2, 3]);
        let data3 = TensorData::new_owned(vec![1, 2, 4]);

        assert_eq!(data1, data2); // Content equality
        assert_ne!(data1, data3);
    }

    #[test]
    fn test_tensor_data_mutable_access() {
        let mut data = TensorData::new_owned(vec![1, 2, 3]);

        if let Some(slice) = data.as_mut_slice() {
            slice[0] = 42;
        }

        assert_eq!(data.as_slice()[0], 42);

        // Shared data should not provide mutable access
        let mut shared_data = TensorData::new_shared(vec![1, 2, 3]);
        assert!(shared_data.as_mut_slice().is_none());
    }
}
