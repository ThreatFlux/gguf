//! Specialized tensor data writing utilities

use crate::error::{GGUFError, Result};
use crate::tensor::{TensorData, TensorInfo};
use std::io::Write;

const MAX_WRITE_CHUNK_SIZE: usize = 1024 * 1024;

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
    /// Preferred buffer size for chunked writing; temporary chunks are capped at 1 MiB.
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
        self.validate_tensor_data(tensor_info, data)?;
        if config.validate_data {
            data.validate()?;
        }

        let data_slice = data.try_as_slice()?;
        self.writer.write_all(data_slice)?;

        let bytes_written = data_slice.len();
        self.advance(bytes_written)?;

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
        let expected_size =
            usize::try_from(tensor_info.checked_expected_data_size()?).map_err(|_| {
                GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' size does not fit this platform",
                    tensor_info.name()
                ))
            })?;
        if expected_size > 0 && config.buffer_size == 0 {
            return Err(GGUFError::InvalidTensorData(
                "Tensor write buffer size must be greater than zero".to_string(),
            ));
        }
        let chunk_size = config.buffer_size.min(expected_size).min(MAX_WRITE_CHUNK_SIZE);
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(chunk_size).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor write chunk".to_string())
        })?;
        buffer.resize(chunk_size, 0);
        let mut total_written = 0;
        let mut checksum = 0u32;

        while total_written < expected_size {
            let to_read = (expected_size - total_written).min(buffer.len());
            reader.read_exact(&mut buffer[..to_read])?;
            self.writer.write_all(&buffer[..to_read])?;

            if config.compute_checksums {
                for (index, &byte) in buffer[..to_read].iter().enumerate() {
                    checksum =
                        checksum.wrapping_add((byte as u32) << ((total_written + index) % 24));
                    checksum = checksum.wrapping_mul(0x9e37_79b9);
                }
            }

            total_written = total_written.checked_add(to_read).ok_or_else(|| {
                GGUFError::InvalidTensorData("Tensor write size overflows usize".to_string())
            })?;
        }

        self.advance(total_written)?;

        Ok(TensorWriteResult {
            bytes_written: total_written,
            position_after: self.position,
            checksum: config.compute_checksums.then_some(checksum),
        })
    }

    /// Validate tensor data before writing
    fn validate_tensor_data(&self, tensor_info: &TensorInfo, data: &TensorData) -> Result<()> {
        let expected_size =
            usize::try_from(tensor_info.checked_expected_data_size()?).map_err(|_| {
                GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' size does not fit this platform",
                    tensor_info.name()
                ))
            })?;
        if data.len() != expected_size {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' size mismatch: expected {}, got {}",
                tensor_info.name(),
                expected_size,
                data.len()
            )));
        }

        Ok(())
    }

    fn advance(&mut self, bytes: usize) -> Result<()> {
        let bytes = u64::try_from(bytes)
            .map_err(|_| GGUFError::InvalidTensorData("Write size does not fit u64".to_string()))?;
        self.position = self.position.checked_add(bytes).ok_or_else(|| {
            GGUFError::InvalidTensorData("Tensor writer position overflows u64".to_string())
        })?;
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
    use crate::tensor::{TensorShape, TensorType};

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

    #[test]
    fn test_zero_chunk_buffer_is_rejected() {
        let mut writer = TensorWriter::new(Vec::new());
        let tensor_info = TensorInfo::new(
            "test".to_string(),
            TensorShape::new(vec![1]).unwrap(),
            TensorType::F32,
            0,
        );
        let config = TensorWriteConfig { buffer_size: 0, ..Default::default() };
        assert!(writer.write_tensor_chunked(&tensor_info, &[][..], &config).is_err());
    }
}
