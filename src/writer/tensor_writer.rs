//! Specialized tensor data writing utilities

use crate::error::{GGUFError, Result};
use crate::tensor::{TensorData, TensorInfo, TensorType};
use std::io::Write;

/// Specialized writer for tensor data
#[derive(Debug)]
pub struct TensorWriter<W> {
    writer: W,
    position: u64,
}

/// Configuration for tensor writing
#[derive(Debug, Clone)]
pub struct TensorWriteConfig {
    /// Whether to validate tensor data before writing
    pub validate_data: bool,
    /// Buffer size for chunked writing
    pub buffer_size: usize,
    /// Whether to compute checksums
    pub compute_checksums: bool,
}

impl Default for TensorWriteConfig {
    fn default() -> Self {
        Self { validate_data: true, buffer_size: 64 * 1024, compute_checksums: false }
    }
}

/// Result of writing tensor data
#[derive(Debug, Clone)]
pub struct TensorWriteResult {
    /// Bytes written
    pub bytes_written: usize,
    /// Position after write
    pub position_after: u64,
    /// Checksum if computed
    pub checksum: Option<u32>,
}

impl<W: Write> TensorWriter<W> {
    /// Create a new tensor writer
    pub fn new(writer: W) -> Self {
        Self { writer, position: 0 }
    }

    /// Write tensor data with default configuration
    pub fn write_tensor(
        &mut self,
        tensor_info: &TensorInfo,
        data: &TensorData,
    ) -> Result<TensorWriteResult> {
        self.write_tensor_with_config(tensor_info, data, &TensorWriteConfig::default())
    }

    /// Write tensor data with custom configuration
    pub fn write_tensor_with_config(
        &mut self,
        tensor_info: &TensorInfo,
        data: &TensorData,
        config: &TensorWriteConfig,
    ) -> Result<TensorWriteResult> {
        if config.validate_data {
            self.validate_tensor_data(tensor_info, data)?;
        }

        let data_slice = data.as_slice();
        self.writer.write_all(data_slice)?;

        let bytes_written = data_slice.len();
        self.position += bytes_written as u64;

        let checksum = if config.compute_checksums { Some(data.checksum()) } else { None };

        Ok(TensorWriteResult { bytes_written, position_after: self.position, checksum })
    }

    /// Write tensor data in chunks
    pub fn write_tensor_chunked<R: std::io::Read>(
        &mut self,
        tensor_info: &TensorInfo,
        mut reader: R,
        config: &TensorWriteConfig,
    ) -> Result<TensorWriteResult> {
        let expected_size = tensor_info.expected_data_size() as usize;
        let mut buffer = vec![0u8; config.buffer_size.min(expected_size)];
        let mut total_written = 0;

        while total_written < expected_size {
            let to_read = (expected_size - total_written).min(buffer.len());
            buffer.resize(to_read, 0);

            reader.read_exact(&mut buffer)?;
            self.writer.write_all(&buffer)?;

            total_written += to_read;
        }

        self.position += total_written as u64;

        Ok(TensorWriteResult {
            bytes_written: total_written,
            position_after: self.position,
            checksum: None, // Not computed for chunked writes
        })
    }

    /// Validate tensor data before writing
    fn validate_tensor_data(&self, tensor_info: &TensorInfo, data: &TensorData) -> Result<()> {
        let expected_size = tensor_info.expected_data_size() as usize;
        if data.len() != expected_size {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' size mismatch: expected {}, got {}",
                tensor_info.name(),
                expected_size,
                data.len()
            )));
        }

        // Type-specific validation
        match tensor_info.tensor_type() {
            TensorType::F32 => {
                if data.len() % 4 != 0 {
                    return Err(GGUFError::InvalidTensorData(
                        "F32 tensor size must be multiple of 4".to_string(),
                    ));
                }
            }
            TensorType::F16 | TensorType::BF16 => {
                if data.len() % 2 != 0 {
                    return Err(GGUFError::InvalidTensorData(
                        "F16/BF16 tensor size must be multiple of 2".to_string(),
                    ));
                }
            }
            _ => {} // Other types don't need specific alignment
        }

        Ok(())
    }

    /// Get current position
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    /// Get the underlying writer
    pub fn into_inner(self) -> W {
        self.writer
    }
}

impl std::fmt::Display for TensorWriteResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorWriteResult {{ bytes: {}, pos: {}{}}}",
            self.bytes_written,
            self.position_after,
            if let Some(checksum) = self.checksum {
                format!(", checksum: 0x{:08x}", checksum)
            } else {
                String::new()
            }
        )
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::tensor::TensorShape;

    #[test]
    fn test_tensor_writer() {
        let buffer = Vec::new();
        let mut writer = TensorWriter::new(buffer);

        let shape = TensorShape::new(vec![2, 2]).unwrap();
        let tensor_info = TensorInfo::new("test".to_string(), shape, TensorType::F32, 0);
        let data = TensorData::new_owned(vec![0u8; 16]);

        let result = writer.write_tensor(&tensor_info, &data).unwrap();
        assert_eq!(result.bytes_written, 16);
        assert_eq!(writer.position(), 16);
    }

    #[test]
    fn test_tensor_validation() {
        let buffer = Vec::new();
        let mut writer = TensorWriter::new(buffer);

        let shape = TensorShape::new(vec![2]).unwrap();
        let tensor_info = TensorInfo::new("test".to_string(), shape, TensorType::F32, 0);

        // Wrong size data
        let wrong_data = TensorData::new_owned(vec![0u8; 4]); // Should be 8
        let result = writer.write_tensor(&tensor_info, &wrong_data);
        assert!(result.is_err());
    }
}
