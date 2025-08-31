//! GGUF file header structures and operations

use crate::error::{GGUFError, Result};
use crate::format::constants::*;
// Simplified implementation to avoid byteorder recursion issues
use crate::format::endian::{read_u32, read_u64, write_u32, write_u64};
use std::io::{Read, Write};

/// GGUF file header structure
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GGUFHeader {
    /// Magic number ("GGUF" in little-endian)
    pub magic: u32,
    /// Version number (currently 3)
    pub version: u32,
    /// Number of tensors in the file
    pub tensor_count: u64,
    /// Number of metadata key-value pairs
    pub metadata_kv_count: u64,
}

impl Default for GGUFHeader {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl GGUFHeader {
    /// Create a new GGUF header with default values
    pub fn new(tensor_count: u64, metadata_kv_count: u64) -> Self {
        Self { magic: GGUF_MAGIC, version: GGUF_VERSION, tensor_count, metadata_kv_count }
    }

    /// Check if the header has valid magic number and version
    pub fn is_valid(&self) -> bool {
        self.magic == GGUF_MAGIC && self.version == GGUF_VERSION
    }

    /// Validate the header and return an error if invalid
    pub fn validate(&self) -> Result<()> {
        if self.magic != GGUF_MAGIC {
            return Err(GGUFError::InvalidMagic { expected: GGUF_MAGIC, found: self.magic });
        }

        if self.version != GGUF_VERSION {
            return Err(GGUFError::UnsupportedVersion(self.version));
        }

        Ok(())
    }

    /// Get the size of the header in bytes
    pub const fn size() -> usize {
        GGUF_HEADER_SIZE
    }

    /// Read a header from a reader
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let magic = read_u32(reader)?;
        let version = read_u32(reader)?;
        let tensor_count = read_u64(reader)?;
        let metadata_kv_count = read_u64(reader)?;

        let header = Self { magic, version, tensor_count, metadata_kv_count };

        header.validate()?;
        Ok(header)
    }

    /// Write the header to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.validate()?;

        write_u32(writer, self.magic)?;
        write_u32(writer, self.version)?;
        write_u64(writer, self.tensor_count)?;
        write_u64(writer, self.metadata_kv_count)?;

        Ok(())
    }

    /// Get the total size of header + metadata + tensor info sections
    pub fn calculate_metadata_size(&self, metadata_size: usize, tensor_info_size: usize) -> usize {
        Self::size() + metadata_size + tensor_info_size
    }

    /// Check if the tensor count is reasonable (not too large)
    pub fn is_tensor_count_reasonable(&self) -> bool {
        // Reasonable limit: 1 million tensors
        self.tensor_count <= 1_000_000
    }

    /// Check if the metadata count is reasonable (not too large)
    pub fn is_metadata_count_reasonable(&self) -> bool {
        // Reasonable limit: 100,000 metadata items
        self.metadata_kv_count <= 100_000
    }

    /// Perform comprehensive validation including reasonableness checks
    pub fn validate_comprehensive(&self) -> Result<()> {
        self.validate()?;

        if !self.is_tensor_count_reasonable() {
            return Err(GGUFError::Format(format!(
                "Unreasonable tensor count: {}",
                self.tensor_count
            )));
        }

        if !self.is_metadata_count_reasonable() {
            return Err(GGUFError::Format(format!(
                "Unreasonable metadata count: {}",
                self.metadata_kv_count
            )));
        }

        Ok(())
    }
}

impl std::fmt::Display for GGUFHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GGUF Header {{ magic: 0x{:08X}, version: {}, tensors: {}, metadata: {} }}",
            self.magic, self.version, self.tensor_count, self.metadata_kv_count
        )
    }
}

/// Information about a tensor in the GGUF file
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Name of the tensor
    pub name: String,
    /// Number of dimensions
    pub n_dimensions: u32,
    /// Shape of the tensor (dimensions)
    pub dimensions: Vec<u64>,
    /// Type of the tensor data
    pub tensor_type: u32,
    /// Offset of the tensor data from the start of the data section
    pub offset: u64,
}

impl TensorInfo {
    /// Create a new tensor info
    pub fn new(name: String, dimensions: Vec<u64>, tensor_type: u32, offset: u64) -> Self {
        let n_dimensions = dimensions.len() as u32;
        Self { name, n_dimensions, dimensions, tensor_type, offset }
    }

    /// Calculate the number of elements in the tensor
    pub fn element_count(&self) -> u64 {
        self.dimensions.iter().product()
    }

    /// Get the shape as a slice
    pub fn shape(&self) -> &[u64] {
        &self.dimensions
    }

    /// Check if the tensor info is valid
    pub fn is_valid(&self) -> bool {
        // Basic validation
        !self.name.is_empty()
            && self.n_dimensions as usize == self.dimensions.len()
            && self.n_dimensions > 0
            && !self.dimensions.is_empty()
            && self.dimensions.iter().all(|&d| d > 0)
    }

    /// Validate the tensor info
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(GGUFError::InvalidTensorData("Tensor name cannot be empty".to_string()));
        }

        if self.n_dimensions as usize != self.dimensions.len() {
            return Err(GGUFError::InvalidTensorData(format!(
                "Dimension count mismatch: expected {}, got {}",
                self.n_dimensions,
                self.dimensions.len()
            )));
        }

        if self.n_dimensions == 0 {
            return Err(GGUFError::InvalidTensorData(
                "Tensor must have at least one dimension".to_string(),
            ));
        }

        if self.dimensions.contains(&0) {
            return Err(GGUFError::InvalidTensorData(
                "All dimensions must be greater than zero".to_string(),
            ));
        }

        // Check for reasonable dimension sizes (prevent integer overflow)
        let max_dim_size = u64::MAX / (self.n_dimensions as u64 * 8); // Conservative limit
        if self.dimensions.iter().any(|&d| d > max_dim_size) {
            return Err(GGUFError::InvalidTensorData("Dimension size too large".to_string()));
        }

        Ok(())
    }

    /// Calculate the minimum size needed to store this tensor info
    pub fn serialized_size(&self) -> usize {
        // String length (8 bytes) + string data + n_dimensions (4 bytes) + dimensions + type (4 bytes) + offset (8 bytes)
        8 + self.name.len() + 4 + (self.dimensions.len() * 8) + 4 + 8
    }

    /// Read tensor info from a reader
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        // Read string length and name
        let name_len = read_u64(reader)? as usize;
        if name_len > GGUF_MAX_STRING_LENGTH {
            return Err(GGUFError::Format(format!("Tensor name too long: {} bytes", name_len)));
        }

        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| GGUFError::Format(format!("Invalid UTF-8 in tensor name: {}", e)))?;

        // Read dimensions
        let n_dimensions = read_u32(reader)?;
        if n_dimensions == 0 || n_dimensions > 8 {
            return Err(GGUFError::InvalidTensorData(format!(
                "Invalid dimension count: {}",
                n_dimensions
            )));
        }

        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(read_u64(reader)?);
        }

        // Read tensor type and offset
        let tensor_type = read_u32(reader)?;
        let offset = read_u64(reader)?;

        let tensor_info = Self { name, n_dimensions, dimensions, tensor_type, offset };

        tensor_info.validate()?;
        Ok(tensor_info)
    }

    /// Write tensor info to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.validate()?;

        // Write string length and name
        write_u64(writer, self.name.len() as u64)?;
        writer.write_all(self.name.as_bytes())?;

        // Write dimensions
        write_u32(writer, self.n_dimensions)?;
        for &dimension in &self.dimensions {
            write_u64(writer, dimension)?;
        }

        // Write tensor type and offset
        write_u32(writer, self.tensor_type)?;
        write_u64(writer, self.offset)?;

        Ok(())
    }
}

impl std::fmt::Display for TensorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorInfo {{ name: '{}', shape: {:?}, type: {}, offset: {} }}",
            self.name, self.dimensions, self.tensor_type, self.offset
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_gguf_header_creation() {
        let header = GGUFHeader::new(10, 5);
        assert_eq!(header.magic, GGUF_MAGIC);
        assert_eq!(header.version, GGUF_VERSION);
        assert_eq!(header.tensor_count, 10);
        assert_eq!(header.metadata_kv_count, 5);
        assert!(header.is_valid());
    }

    #[test]
    fn test_gguf_header_validation() {
        let mut header = GGUFHeader::new(10, 5);
        assert!(header.validate().is_ok());

        header.magic = 0x12345678;
        assert!(header.validate().is_err());

        header.magic = GGUF_MAGIC;
        header.version = 999;
        assert!(header.validate().is_err());
    }

    #[test]
    fn test_gguf_header_io() {
        let original = GGUFHeader::new(42, 13);

        // Write to buffer
        let mut buffer = Vec::new();
        original.write_to(&mut buffer).unwrap();

        // Read back from buffer
        let mut cursor = Cursor::new(buffer);
        let read_header = GGUFHeader::read_from(&mut cursor).unwrap();

        assert_eq!(original, read_header);
    }

    #[test]
    fn test_tensor_info_creation() {
        let tensor_info = TensorInfo::new(
            "test_tensor".to_string(),
            vec![2, 3, 4],
            1, // F16
            1024,
        );

        assert_eq!(tensor_info.name, "test_tensor");
        assert_eq!(tensor_info.n_dimensions, 3);
        assert_eq!(tensor_info.dimensions, vec![2, 3, 4]);
        assert_eq!(tensor_info.tensor_type, 1);
        assert_eq!(tensor_info.offset, 1024);
        assert_eq!(tensor_info.element_count(), 24);
        assert!(tensor_info.is_valid());
    }

    #[test]
    fn test_tensor_info_validation() {
        // Valid tensor info
        let valid = TensorInfo::new("test".to_string(), vec![2, 3], 0, 0);
        assert!(valid.validate().is_ok());

        // Empty name
        let empty_name = TensorInfo::new("".to_string(), vec![2], 0, 0);
        assert!(empty_name.validate().is_err());

        // Zero dimension
        let zero_dim = TensorInfo::new("test".to_string(), vec![2, 0], 0, 0);
        assert!(zero_dim.validate().is_err());

        // Mismatched dimension count
        let mut mismatched = TensorInfo::new("test".to_string(), vec![2, 3], 0, 0);
        mismatched.n_dimensions = 5;
        assert!(mismatched.validate().is_err());
    }

    #[test]
    fn test_tensor_info_io() {
        let original = TensorInfo::new("test_tensor".to_string(), vec![2, 3, 4], 1, 1024);

        // Write to buffer
        let mut buffer = Vec::new();
        original.write_to(&mut buffer).unwrap();

        // Read back from buffer
        let mut cursor = Cursor::new(buffer);
        let read_info = TensorInfo::read_from(&mut cursor).unwrap();

        assert_eq!(original, read_info);
    }

    #[test]
    fn test_header_reasonableness_checks() {
        let mut header = GGUFHeader::new(10, 5);
        assert!(header.validate_comprehensive().is_ok());

        header.tensor_count = 2_000_000; // Too many tensors
        assert!(header.validate_comprehensive().is_err());

        header.tensor_count = 10;
        header.metadata_kv_count = 200_000; // Too much metadata
        assert!(header.validate_comprehensive().is_err());
    }

    #[test]
    fn test_header_display() {
        let header = GGUFHeader::new(42, 13);
        let display_str = format!("{}", header);
        assert!(display_str.contains("42"));
        assert!(display_str.contains("13"));
        assert!(display_str.contains("GGUF Header"));
    }

    #[test]
    fn test_tensor_info_display() {
        let info = TensorInfo::new("test".to_string(), vec![2, 3], 1, 1024);
        let display_str = format!("{}", info);
        assert!(display_str.contains("test"));
        assert!(display_str.contains("[2, 3]"));
        assert!(display_str.contains("1024"));
    }
}
