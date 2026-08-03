//! Tensor builder utilities

use crate::error::{GGUFError, Result};
use crate::tensor::{TensorData, TensorInfo, TensorShape, TensorType};

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use hashbrown::HashSet;
#[cfg(feature = "std")]
use std::collections::HashSet;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

/// Builder for tensor collections
#[cfg(any(feature = "std", feature = "alloc"))]
#[derive(Debug, Default)]
pub struct TensorCollectionBuilder {
    tensors: Vec<(TensorInfo, TensorData)>,
    names: HashSet<String>,
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
        Self { tensors: Vec::new(), names: HashSet::new() }
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
        if self.names.contains(&name) {
            return Err(GGUFError::InvalidTensorData(format!("Duplicate tensor name: '{}'", name)));
        }
        let shape = TensorShape::new(shape)?;
        let tensor_info = TensorInfo::new(name.clone(), shape, tensor_type, 0);
        let tensor_data = TensorData::new_owned(data);

        // Validate size
        let expected_size =
            usize::try_from(tensor_info.checked_expected_data_size()?).map_err(|_| {
                GGUFError::InvalidTensorData("Tensor size does not fit this platform".to_string())
            })?;
        if tensor_data.len() != expected_size {
            return Err(GGUFError::InvalidTensorData(format!(
                "Size mismatch for tensor '{}'",
                name
            )));
        }

        tensor_info.validate()?;
        tensor_data.validate()?;
        self.tensors.try_reserve(1).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor collection".to_string())
        })?;
        self.names.try_reserve(1).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor-name index".to_string())
        })?;
        self.names.insert(name);
        self.tensors.push((tensor_info, tensor_data));
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
        if self.names.contains(&name) {
            return Err(GGUFError::InvalidTensorData(format!("Duplicate tensor name: '{}'", name)));
        }
        let shape = TensorShape::new(shape)?;
        let tensor_info = TensorInfo::new(name.clone(), shape, tensor_type, 0);

        // Validate size
        let expected_size =
            usize::try_from(tensor_info.checked_expected_data_size()?).map_err(|_| {
                GGUFError::InvalidTensorData("Tensor size does not fit this platform".to_string())
            })?;
        if data.len() != expected_size {
            return Err(GGUFError::InvalidTensorData(format!(
                "Size mismatch for tensor '{}'",
                name
            )));
        }

        tensor_info.validate()?;
        data.validate()?;
        self.tensors.try_reserve(1).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor collection".to_string())
        })?;
        self.names.try_reserve(1).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor-name index".to_string())
        })?;
        self.names.insert(name);
        self.tensors.push((tensor_info, data));
        Ok(self)
    }

    /// Build the tensor collection
    pub fn build(self) -> Vec<(TensorInfo, TensorData)> {
        self.tensors
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
        self.names.contains(name)
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
    #[cfg(any(feature = "std", feature = "alloc"))]
    fn validate_pair(
        tensor_info: TensorInfo,
        tensor_data: TensorData,
    ) -> Result<(TensorInfo, TensorData)> {
        tensor_info.validate()?;
        tensor_data.validate()?;
        let expected_size =
            usize::try_from(tensor_info.checked_expected_data_size()?).map_err(|_| {
                GGUFError::InvalidTensorData("Tensor size does not fit this platform".to_string())
            })?;
        if tensor_data.len() != expected_size {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' data size mismatch: expected {}, got {}",
                tensor_info.name(),
                expected_size,
                tensor_data.len()
            )));
        }
        Ok((tensor_info, tensor_data))
    }

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
        Self::validate_pair(tensor_info, tensor_data)
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
        Self::validate_pair(tensor_info, tensor_data)
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
        Self::weight_matrix(name, embedding_dim, vocab_size, tensor_type, data)
    }
}

#[cfg(all(test, feature = "std"))]
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

    #[test]
    fn test_embedding_matrix_uses_ggml_dimension_order() {
        let (info, _) = TensorPatterns::embedding_matrix(
            "token_embd.weight".to_string(),
            32_000,
            2,
            TensorType::F16,
            vec![0; 32_000 * 2 * 2],
        )
        .unwrap();

        assert_eq!(info.shape().dims(), &[2, 32_000]);
    }

    #[test]
    fn test_tensor_collection_rejects_duplicates_and_preserves_order() {
        let duplicate = TensorCollectionBuilder::new()
            .add_tensor("first", vec![1], TensorType::F32, vec![0; 4])
            .unwrap()
            .add_tensor("first", vec![1], TensorType::F32, vec![0; 4]);
        assert!(duplicate.is_err());

        let collection = TensorCollectionBuilder::new()
            .add_tensor("first", vec![1], TensorType::F32, vec![0; 4])
            .unwrap()
            .add_tensor("second", vec![1], TensorType::F32, vec![0; 4])
            .unwrap()
            .build();
        assert_eq!(collection[0].0.name(), "first");
        assert_eq!(collection[1].0.name(), "second");
    }

    #[test]
    fn test_tensor_patterns_reject_wrong_payload_size() {
        assert!(TensorPatterns::weight_matrix(
            "bad".to_string(),
            2,
            3,
            TensorType::F32,
            vec![0; 4],
        )
        .is_err());
        assert!(TensorPatterns::bias_vector("bad".to_string(), 3, TensorType::F32, vec![0; 4],)
            .is_err());
    }
}
