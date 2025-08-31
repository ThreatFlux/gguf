//! Tensor shape and dimension handling

use crate::error::{GGUFError, Result};
#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

/// Represents the shape of a tensor
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorShape {
    /// Dimensions of the tensor
    pub dimensions: Vec<u64>,
}

impl TensorShape {
    /// Create a new tensor shape from dimensions
    pub fn new(dimensions: Vec<u64>) -> Result<Self> {
        if dimensions.is_empty() {
            return Err(GGUFError::InvalidTensorData("Tensor shape cannot be empty".to_string()));
        }

        if dimensions.contains(&0) {
            return Err(GGUFError::InvalidTensorData(
                "Tensor dimensions cannot be zero".to_string(),
            ));
        }

        // Check for reasonable dimension sizes to prevent overflow
        const MAX_REASONABLE_DIM: u64 = 1_000_000_000; // 1B elements per dimension
        if dimensions.iter().any(|&d| d > MAX_REASONABLE_DIM) {
            return Err(GGUFError::InvalidTensorData("Tensor dimension too large".to_string()));
        }

        Ok(Self { dimensions })
    }

    /// Create a tensor shape without validation (unsafe)
    pub fn new_unchecked(dimensions: Vec<u64>) -> Self {
        Self { dimensions }
    }

    /// Create a scalar tensor (1D with size 1)
    pub fn scalar() -> Self {
        Self::new_unchecked(vec![1])
    }

    /// Create a vector tensor (1D)
    pub fn vector(size: u64) -> Result<Self> {
        Self::new(vec![size])
    }

    /// Create a matrix tensor (2D)
    pub fn matrix(rows: u64, cols: u64) -> Result<Self> {
        Self::new(vec![rows, cols])
    }

    /// Create a 3D tensor
    pub fn tensor_3d(dim0: u64, dim1: u64, dim2: u64) -> Result<Self> {
        Self::new(vec![dim0, dim1, dim2])
    }

    /// Create a 4D tensor
    pub fn tensor_4d(dim0: u64, dim1: u64, dim2: u64, dim3: u64) -> Result<Self> {
        Self::new(vec![dim0, dim1, dim2, dim3])
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.dimensions.len()
    }

    /// Get the dimensions as a slice
    pub fn dims(&self) -> &[u64] {
        &self.dimensions
    }

    /// Get the dimensions as a mutable slice
    pub fn dims_mut(&mut self) -> &mut [u64] {
        &mut self.dimensions
    }

    /// Get a specific dimension
    pub fn dim(&self, index: usize) -> Option<u64> {
        self.dimensions.get(index).copied()
    }

    /// Calculate the total number of elements
    pub fn element_count(&self) -> u64 {
        self.dimensions
            .iter()
            .try_fold(1u64, |acc, &dim| acc.checked_mul(dim).ok_or(()))
            .unwrap_or(u64::MAX)
    }

    /// Check if this is a scalar (single element)
    pub fn is_scalar(&self) -> bool {
        self.element_count() == 1
    }

    /// Check if this is a vector (1D with size > 1)
    pub fn is_vector(&self) -> bool {
        self.ndim() == 1 && self.dimensions[0] > 1
    }

    /// Check if this is a matrix (2D)
    pub fn is_matrix(&self) -> bool {
        self.ndim() == 2
    }

    /// Check if this is a 3D tensor
    pub fn is_3d(&self) -> bool {
        self.ndim() == 3
    }

    /// Check if this is a 4D tensor
    pub fn is_4d(&self) -> bool {
        self.ndim() == 4
    }

    /// Check if this tensor is contiguous (for certain operations)
    pub fn is_contiguous(&self) -> bool {
        // A tensor is considered contiguous if its strides follow the default pattern
        // For simplicity, we'll consider all tensors contiguous here
        true
    }

    /// Calculate strides for C-style (row-major) ordering
    pub fn calculate_strides(&self) -> Vec<u64> {
        let mut strides = vec![1u64; self.ndim()];

        if self.ndim() > 1 {
            for i in (0..self.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * self.dimensions[i + 1];
            }
        }

        strides
    }

    /// Calculate the memory layout size with strides
    pub fn memory_size(&self) -> u64 {
        if self.dimensions.is_empty() {
            return 0;
        }

        let strides = self.calculate_strides();
        strides[0] * self.dimensions[0]
    }

    /// Reshape the tensor to a new shape with the same number of elements
    pub fn reshape(&self, new_dimensions: Vec<u64>) -> Result<TensorShape> {
        let new_shape = TensorShape::new(new_dimensions)?;

        if self.element_count() != new_shape.element_count() {
            return Err(GGUFError::InvalidTensorData(format!(
                "Cannot reshape tensor with {} elements to shape with {} elements",
                self.element_count(),
                new_shape.element_count()
            )));
        }

        Ok(new_shape)
    }

    /// Flatten the tensor to 1D
    pub fn flatten(&self) -> TensorShape {
        TensorShape::new_unchecked(vec![self.element_count()])
    }

    /// Transpose the tensor (swap first two dimensions)
    pub fn transpose(&self) -> Result<TensorShape> {
        if self.ndim() < 2 {
            return Err(GGUFError::InvalidTensorData(
                "Cannot transpose tensor with less than 2 dimensions".to_string(),
            ));
        }

        let mut new_dims = self.dimensions.clone();
        new_dims.swap(0, 1);

        Ok(TensorShape::new_unchecked(new_dims))
    }

    /// Add a dimension at the beginning (unsqueeze at dim 0)
    pub fn unsqueeze_front(&self) -> TensorShape {
        let mut new_dims = vec![1];
        new_dims.extend_from_slice(&self.dimensions);
        TensorShape::new_unchecked(new_dims)
    }

    /// Add a dimension at the end (unsqueeze at last dim)
    pub fn unsqueeze_back(&self) -> TensorShape {
        let mut new_dims = self.dimensions.clone();
        new_dims.push(1);
        TensorShape::new_unchecked(new_dims)
    }

    /// Remove dimensions of size 1
    pub fn squeeze(&self) -> TensorShape {
        let squeezed_dims: Vec<u64> = self.dimensions.iter().copied().filter(|&d| d != 1).collect();

        if squeezed_dims.is_empty() {
            // If all dimensions were 1, keep one dimension
            TensorShape::new_unchecked(vec![1])
        } else {
            TensorShape::new_unchecked(squeezed_dims)
        }
    }

    /// Check if two shapes are broadcastable
    pub fn is_broadcastable_with(&self, other: &TensorShape) -> bool {
        let max_ndim = self.ndim().max(other.ndim());

        for i in 0..max_ndim {
            let dim_a = if i < self.ndim() { self.dimensions[self.ndim() - 1 - i] } else { 1 };

            let dim_b = if i < other.ndim() { other.dimensions[other.ndim() - 1 - i] } else { 1 };

            if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
                return false;
            }
        }

        true
    }

    /// Calculate the broadcasted shape with another shape
    pub fn broadcast_with(&self, other: &TensorShape) -> Result<TensorShape> {
        if !self.is_broadcastable_with(other) {
            return Err(GGUFError::InvalidTensorData(format!(
                "Shapes {:?} and {:?} are not broadcastable",
                self.dimensions, other.dimensions
            )));
        }

        let max_ndim = self.ndim().max(other.ndim());
        let mut result_dims = Vec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let dim_a = if i < self.ndim() { self.dimensions[self.ndim() - 1 - i] } else { 1 };

            let dim_b = if i < other.ndim() { other.dimensions[other.ndim() - 1 - i] } else { 1 };

            result_dims.push(dim_a.max(dim_b));
        }

        result_dims.reverse();
        Ok(TensorShape::new_unchecked(result_dims))
    }

    /// Get the shape for matrix multiplication (if applicable)
    pub fn matmul_output_shape(&self, other: &TensorShape) -> Result<TensorShape> {
        if !self.is_matrix() || !other.is_matrix() {
            return Err(GGUFError::InvalidTensorData(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        if self.dimensions[1] != other.dimensions[0] {
            return Err(GGUFError::InvalidTensorData(format!(
                "Cannot multiply matrices with shapes {:?} and {:?}: dimension mismatch",
                self.dimensions, other.dimensions
            )));
        }

        TensorShape::matrix(self.dimensions[0], other.dimensions[1])
    }

    /// Check if this shape is compatible for element-wise operations
    pub fn is_elementwise_compatible(&self, other: &TensorShape) -> bool {
        self == other || self.is_broadcastable_with(other)
    }

    /// Get a string representation of the shape
    pub fn shape_string(&self) -> String {
        if self.dimensions.len() == 1 {
            format!("({})", self.dimensions[0])
        } else {
            format!(
                "({})",
                self.dimensions.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
            )
        }
    }

    /// Validate that the shape is reasonable for practical use
    pub fn validate_practical_limits(&self) -> Result<()> {
        // Check for reasonable total elements (prevent memory issues)
        const MAX_ELEMENTS: u64 = u32::MAX as u64; // ~4B elements max
        if self.element_count() > MAX_ELEMENTS {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor too large: {} elements exceeds practical limit of {}",
                self.element_count(),
                MAX_ELEMENTS
            )));
        }

        // Check for reasonable number of dimensions
        const MAX_DIMENSIONS: usize = 8;
        if self.ndim() > MAX_DIMENSIONS {
            return Err(GGUFError::InvalidTensorData(format!(
                "Too many dimensions: {} exceeds practical limit of {}",
                self.ndim(),
                MAX_DIMENSIONS
            )));
        }

        Ok(())
    }
}

impl std::fmt::Display for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.shape_string())
    }
}

impl From<Vec<u64>> for TensorShape {
    fn from(dimensions: Vec<u64>) -> Self {
        TensorShape::new(dimensions).expect("Invalid tensor shape")
    }
}

impl From<&[u64]> for TensorShape {
    fn from(dimensions: &[u64]) -> Self {
        TensorShape::new(dimensions.to_vec()).expect("Invalid tensor shape")
    }
}

impl AsRef<[u64]> for TensorShape {
    fn as_ref(&self) -> &[u64] {
        &self.dimensions
    }
}

impl std::ops::Index<usize> for TensorShape {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dimensions[index]
    }
}

impl std::ops::IndexMut<usize> for TensorShape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.dimensions[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape_creation() {
        let shape = TensorShape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.element_count(), 24);
    }

    #[test]
    fn test_tensor_shape_validation() {
        assert!(TensorShape::new(vec![]).is_err()); // Empty
        assert!(TensorShape::new(vec![0, 3]).is_err()); // Zero dimension
        assert!(TensorShape::new(vec![2, 3]).is_ok());
    }

    #[test]
    fn test_tensor_shape_types() {
        let scalar = TensorShape::scalar();
        assert!(scalar.is_scalar());
        assert!(!scalar.is_vector());

        let vector = TensorShape::vector(10).unwrap();
        assert!(vector.is_vector());
        assert!(!vector.is_matrix());

        let matrix = TensorShape::matrix(3, 4).unwrap();
        assert!(matrix.is_matrix());
        assert!(!matrix.is_3d());
    }

    #[test]
    fn test_strides_calculation() {
        let shape = TensorShape::new(vec![2, 3, 4]).unwrap();
        let strides = shape.calculate_strides();
        assert_eq!(strides, vec![12, 4, 1]); // C-style ordering
    }

    #[test]
    fn test_reshape() {
        let shape = TensorShape::new(vec![2, 3, 4]).unwrap();

        // Valid reshape
        let reshaped = shape.reshape(vec![6, 4]).unwrap();
        assert_eq!(reshaped.dims(), &[6, 4]);
        assert_eq!(reshaped.element_count(), 24);

        // Invalid reshape (different element count)
        assert!(shape.reshape(vec![2, 5]).is_err());
    }

    #[test]
    fn test_transpose() {
        let shape = TensorShape::matrix(3, 4).unwrap();
        let transposed = shape.transpose().unwrap();
        assert_eq!(transposed.dims(), &[4, 3]);

        let vector = TensorShape::vector(5).unwrap();
        assert!(vector.transpose().is_err()); // Cannot transpose 1D
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let shape = TensorShape::new(vec![1, 3, 1, 4]).unwrap();
        let squeezed = shape.squeeze();
        assert_eq!(squeezed.dims(), &[3, 4]);

        let unsqueezed = squeezed.unsqueeze_front();
        assert_eq!(unsqueezed.dims(), &[1, 3, 4]);
    }

    #[test]
    fn test_broadcasting() {
        let shape1 = TensorShape::new(vec![3, 1, 4]).unwrap();
        let shape2 = TensorShape::new(vec![2, 4]).unwrap();

        assert!(shape1.is_broadcastable_with(&shape2));

        let broadcasted = shape1.broadcast_with(&shape2).unwrap();
        assert_eq!(broadcasted.dims(), &[3, 2, 4]);
    }

    #[test]
    fn test_matmul_shape() {
        let shape1 = TensorShape::matrix(3, 4).unwrap();
        let shape2 = TensorShape::matrix(4, 5).unwrap();

        let result = shape1.matmul_output_shape(&shape2).unwrap();
        assert_eq!(result.dims(), &[3, 5]);

        // Invalid matmul
        let shape3 = TensorShape::matrix(3, 7).unwrap();
        assert!(shape1.matmul_output_shape(&shape3).is_err());
    }

    #[test]
    fn test_practical_limits() {
        let reasonable = TensorShape::new(vec![100, 100]).unwrap();
        assert!(reasonable.validate_practical_limits().is_ok());

        // This would be too large in practice
        let huge_dims = vec![100_000; 10];
        if let Ok(huge_shape) = TensorShape::new(huge_dims) {
            assert!(huge_shape.validate_practical_limits().is_err());
        }
    }

    #[test]
    fn test_elementwise_compatibility() {
        let shape1 = TensorShape::new(vec![3, 4]).unwrap();
        let shape2 = TensorShape::new(vec![3, 4]).unwrap();
        let shape3 = TensorShape::new(vec![1, 4]).unwrap();
        let shape4 = TensorShape::new(vec![2, 3]).unwrap();

        assert!(shape1.is_elementwise_compatible(&shape2)); // Same shape
        assert!(shape1.is_elementwise_compatible(&shape3)); // Broadcastable
        assert!(!shape1.is_elementwise_compatible(&shape4)); // Not compatible
    }

    #[test]
    fn test_shape_display() {
        let shape = TensorShape::new(vec![2, 3, 4]).unwrap();
        let display_str = format!("{}", shape);
        assert!(display_str.contains("2"));
        assert!(display_str.contains("3"));
        assert!(display_str.contains("4"));
    }

    #[test]
    fn test_shape_indexing() {
        let shape = TensorShape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(shape[0], 2);
        assert_eq!(shape[1], 3);
        assert_eq!(shape[2], 4);
    }

    #[test]
    fn test_shape_conversions() {
        let vec_dims = vec![2, 3, 4];
        let shape1: TensorShape = vec_dims.clone().into();
        let shape2: TensorShape = vec_dims.as_slice().into();

        assert_eq!(shape1.dims(), &[2, 3, 4]);
        assert_eq!(shape2.dims(), &[2, 3, 4]);

        let as_ref: &[u64] = shape1.as_ref();
        assert_eq!(as_ref, &[2, 3, 4]);
    }
}
