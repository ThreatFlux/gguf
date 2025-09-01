//! Tensor builder utilities

use crate::error::{GGUFError, Result};
use crate::tensor::{TensorData, TensorInfo, TensorShape, TensorType};

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::{format, string::String, vec, vec::Vec};

/// Builder for tensor collections
#[cfg(any(feature = "std", feature = "alloc"))]
#[derive(Debug, Default)]
pub struct TensorCollectionBuilder {
    tensors: HashMap<String, (TensorInfo, TensorData)>,
}

/// Builder for tensor collections (no_std + no_alloc variant)
#[cfg(not(any(feature = "std", feature = "alloc")))]
#[derive(Debug, Default)]
pub struct TensorCollectionBuilder {
    // Placeholder for no_std + no_alloc builds
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl TensorCollectionBuilder {
    /// Create a new tensor collection builder
    pub fn new() -> Self {
        Self { tensors: HashMap::new() }
    }

    /// Add a tensor
    pub fn add_tensor<N: Into<String>>(
        mut self,
        name: N,
        shape: Vec<u64>,
        tensor_type: TensorType,
        data: Vec<u8>,
    ) -> Result<Self> {
        let name = name.into();
        let shape = TensorShape::new(shape)?;
        let tensor_info = TensorInfo::new(name.clone(), shape, tensor_type, 0);
        let tensor_data = TensorData::new_owned(data);

        // Validate size
        if tensor_data.len() != tensor_info.expected_data_size() as usize {
            return Err(GGUFError::InvalidTensorData(format!(
                "Size mismatch for tensor '{}'",
                name
            )));
        }

        self.tensors.insert(name, (tensor_info, tensor_data));
        Ok(self)
    }

    /// Add tensor with TensorData
    pub fn add_tensor_data<N: Into<String>>(
        mut self,
        name: N,
        shape: Vec<u64>,
        tensor_type: TensorType,
        data: TensorData,
    ) -> Result<Self> {
        let name = name.into();
        let shape = TensorShape::new(shape)?;
        let tensor_info = TensorInfo::new(name.clone(), shape, tensor_type, 0);

        // Validate size
        if data.len() != tensor_info.expected_data_size() as usize {
            return Err(GGUFError::InvalidTensorData(format!(
                "Size mismatch for tensor '{}'",
                name
            )));
        }

        self.tensors.insert(name, (tensor_info, data));
        Ok(self)
    }

    /// Build the tensor collection
    pub fn build(self) -> Vec<(TensorInfo, TensorData)> {
        self.tensors.into_values().collect()
    }

    /// Get tensor count
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Check if tensor exists
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

#[cfg(not(any(feature = "std", feature = "alloc")))]
impl TensorCollectionBuilder {
    /// Create a new tensor collection builder (no-op for no_std + no_alloc)
    pub fn new() -> Self {
        Self {}
    }

    /// Add a tensor (returns error for no_std + no_alloc)
    pub fn add_tensor<N>(
        self,
        _name: N,
        _shape: &[u64],
        _tensor_type: TensorType,
        _data: &[u8],
    ) -> Result<Self> {
        Err(GGUFError::AllocationRequired)
    }

    /// Add tensor with TensorData (returns error for no_std + no_alloc)
    pub fn add_tensor_data<N>(
        self,
        _name: N,
        _shape: &[u64],
        _tensor_type: TensorType,
        _data: &TensorData,
    ) -> Result<Self> {
        Err(GGUFError::AllocationRequired)
    }

    /// Build the tensor collection (returns empty for no_std + no_alloc)
    pub fn build(self) -> &'static [(TensorInfo, TensorData)] {
        &[]
    }

    /// Get tensor count (always 0 for no_std + no_alloc)
    pub fn len(&self) -> usize {
        0
    }

    /// Check if empty (always true for no_std + no_alloc)
    pub fn is_empty(&self) -> bool {
        true
    }

    /// Check if tensor exists (always false for no_std + no_alloc)
    pub fn contains(&self, _name: &str) -> bool {
        false
    }
}

/// Helper for creating common tensor patterns
pub struct TensorPatterns;

impl TensorPatterns {
    /// Create a weight matrix tensor
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn weight_matrix(
        name: String,
        input_dim: u64,
        output_dim: u64,
        tensor_type: TensorType,
        data: Vec<u8>,
    ) -> Result<(TensorInfo, TensorData)> {
        let shape = TensorShape::new(vec![input_dim, output_dim])?;
        let tensor_info = TensorInfo::new(name, shape, tensor_type, 0);
        let tensor_data = TensorData::new_owned(data);
        Ok((tensor_info, tensor_data))
    }

    /// Create a bias vector tensor
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn bias_vector(
        name: String,
        dim: u64,
        tensor_type: TensorType,
        data: Vec<u8>,
    ) -> Result<(TensorInfo, TensorData)> {
        let shape = TensorShape::new(vec![dim])?;
        let tensor_info = TensorInfo::new(name, shape, tensor_type, 0);
        let tensor_data = TensorData::new_owned(data);
        Ok((tensor_info, tensor_data))
    }

    /// Create an embedding matrix
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn embedding_matrix(
        name: String,
        vocab_size: u64,
        embedding_dim: u64,
        tensor_type: TensorType,
        data: Vec<u8>,
    ) -> Result<(TensorInfo, TensorData)> {
        Self::weight_matrix(name, vocab_size, embedding_dim, tensor_type, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_collection_builder() {
        let collection = TensorCollectionBuilder::new()
            .add_tensor("weight", vec![2, 3], TensorType::F32, vec![0u8; 24])
            .unwrap()
            .add_tensor("bias", vec![3], TensorType::F32, vec![0u8; 12])
            .unwrap()
            .build();

        assert_eq!(collection.len(), 2);
    }

    #[test]
    fn test_tensor_patterns() {
        let (info, data) = TensorPatterns::weight_matrix(
            "test_weight".to_string(),
            4,
            3,
            TensorType::F32,
            vec![0u8; 48], // 4*3*4 bytes
        )
        .unwrap();

        assert_eq!(info.name(), "test_weight");
        assert_eq!(info.shape().dims(), &[4, 3]);
        assert_eq!(data.len(), 48);
    }
}
