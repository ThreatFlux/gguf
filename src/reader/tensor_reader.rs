//! Tensor-specific reading utilities

use crate::error::{GGUFError, Result};
use crate::tensor::{quantization::QuantizationParams, TensorData, TensorInfo, TensorType};
use std::io::{Read, Seek, SeekFrom};

/// Specialized reader for tensor data with format-specific handling
#[derive(Debug)]
pub struct TensorReader<R> {
    /// Underlying reader
    reader: R,
    /// Current position
    position: u64,
}

/// Options for tensor reading
#[derive(Debug, Clone)]
pub struct TensorReadOptions {
    /// Whether to validate data alignment
    pub validate_alignment: bool,
    /// Whether to perform data integrity checks
    pub validate_integrity: bool,
    /// Whether to decompress quantized data
    pub decompress_quantized: bool,
    /// Maximum tensor size to read (0 = no limit)
    pub max_tensor_size: usize,
    /// Buffer size for reading
    pub buffer_size: usize,
}

impl Default for TensorReadOptions {
    fn default() -> Self {
        Self {
            validate_alignment: true,
            validate_integrity: false,   // Can be expensive
            decompress_quantized: false, // Keep raw quantized data by default
            max_tensor_size: 0,
            buffer_size: 1024 * 1024, // 1MB buffer
        }
    }
}

/// Result of reading tensor data
#[derive(Debug, Clone)]
pub struct TensorReadResult {
    /// The tensor data
    pub data: TensorData,
    /// Actual bytes read
    pub bytes_read: usize,
    /// Whether the data was validated
    pub was_validated: bool,
    /// Whether the data was decompressed
    pub was_decompressed: bool,
    /// Checksum of the data (if computed)
    pub checksum: Option<u32>,
}

impl<R: Read> TensorReader<R> {
    /// Create a new tensor reader
    pub fn new(reader: R) -> Self {
        Self { reader, position: 0 }
    }

    /// Read tensor data with default options
    pub fn read_tensor_data(&mut self, tensor_info: &TensorInfo) -> Result<TensorReadResult> {
        self.read_tensor_data_with_options(tensor_info, &TensorReadOptions::default())
    }

    /// Read tensor data with custom options
    pub fn read_tensor_data_with_options(
        &mut self,
        tensor_info: &TensorInfo,
        options: &TensorReadOptions,
    ) -> Result<TensorReadResult> {
        let expected_size = tensor_info.expected_data_size() as usize;

        // Check size limits
        if options.max_tensor_size > 0 && expected_size > options.max_tensor_size {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' size {} exceeds maximum {}",
                tensor_info.name(),
                expected_size,
                options.max_tensor_size
            )));
        }

        // Read the raw data
        let mut data = vec![0u8; expected_size];
        self.reader.read_exact(&mut data)?;
        self.position += expected_size as u64;

        let mut tensor_data = TensorData::new_owned(data);
        let mut was_validated = false;
        let mut was_decompressed = false;
        let mut checksum = None;

        // Validate alignment if requested
        if options.validate_alignment {
            self.validate_tensor_alignment(tensor_info, &tensor_data)?;
        }

        // Validate integrity if requested
        if options.validate_integrity {
            checksum = Some(tensor_data.checksum());
            self.validate_tensor_integrity(tensor_info, &tensor_data)?;
            was_validated = true;
        }

        // Decompress quantized data if requested
        if options.decompress_quantized && tensor_info.tensor_type().is_quantized() {
            tensor_data = self.decompress_quantized_tensor(tensor_info, tensor_data)?;
            was_decompressed = true;
        }

        Ok(TensorReadResult {
            data: tensor_data,
            bytes_read: expected_size,
            was_validated,
            was_decompressed,
            checksum,
        })
    }

    /// Read multiple tensors efficiently
    pub fn read_multiple_tensors(
        &mut self,
        tensor_infos: &[&TensorInfo],
        options: &TensorReadOptions,
    ) -> Result<Vec<TensorReadResult>> {
        let mut results = Vec::with_capacity(tensor_infos.len());

        for tensor_info in tensor_infos {
            let result = self.read_tensor_data_with_options(tensor_info, options)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Read tensor data in chunks for large tensors
    pub fn read_tensor_chunked(
        &mut self,
        tensor_info: &TensorInfo,
        chunk_size: usize,
        mut callback: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        let total_size = tensor_info.expected_data_size() as usize;
        let mut remaining = total_size;
        let mut buffer = vec![0u8; chunk_size.min(remaining)];

        while remaining > 0 {
            let to_read = chunk_size.min(remaining);
            buffer.resize(to_read, 0);

            self.reader.read_exact(&mut buffer)?;
            self.position += to_read as u64;

            callback(&buffer)?;
            remaining -= to_read;
        }

        Ok(())
    }

    /// Validate tensor data alignment
    fn validate_tensor_alignment(&self, tensor_info: &TensorInfo, data: &TensorData) -> Result<()> {
        let required_alignment = tensor_info.tensor_type().element_size();

        if !data.is_aligned_to(required_alignment) {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' data not properly aligned to {} bytes",
                tensor_info.name(),
                required_alignment
            )));
        }

        Ok(())
    }

    /// Validate tensor data integrity
    fn validate_tensor_integrity(&self, tensor_info: &TensorInfo, data: &TensorData) -> Result<()> {
        let expected_size = tensor_info.expected_data_size() as usize;
        if data.len() != expected_size {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' size mismatch: expected {} bytes, got {}",
                tensor_info.name(),
                expected_size,
                data.len()
            )));
        }

        // Additional validation based on tensor type
        match tensor_info.tensor_type() {
            TensorType::F32 => {
                if data.len() % 4 != 0 {
                    return Err(GGUFError::InvalidTensorData(format!(
                        "F32 tensor '{}' size not multiple of 4 bytes",
                        tensor_info.name()
                    )));
                }
            }
            TensorType::F16 | TensorType::BF16 => {
                if data.len() % 2 != 0 {
                    return Err(GGUFError::InvalidTensorData(format!(
                        "F16/BF16 tensor '{}' size not multiple of 2 bytes",
                        tensor_info.name()
                    )));
                }
            }
            _ => {
                // For quantized types, validate block alignment
                if tensor_info.tensor_type().is_quantized() {
                    let params = QuantizationParams::for_type(tensor_info.tensor_type());
                    let expected_blocks = params.calculate_num_blocks(tensor_info.element_count());
                    let expected_size = expected_blocks * params.block_size_bytes as u64;

                    if data.len() != expected_size as usize {
                        return Err(GGUFError::InvalidTensorData(format!(
                            "Quantized tensor '{}' block size mismatch",
                            tensor_info.name()
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Decompress quantized tensor data (placeholder implementation)
    fn decompress_quantized_tensor(
        &self,
        tensor_info: &TensorInfo,
        data: TensorData,
    ) -> Result<TensorData> {
        // This is a placeholder - actual quantization decompression would require
        // implementing the specific algorithms for each quantization type

        match tensor_info.tensor_type() {
            TensorType::Q4_0 | TensorType::Q4_1 => {
                // Would implement Q4 decompression to F32
                Ok(data) // For now, return unchanged
            }
            TensorType::Q8_0 | TensorType::Q8_1 => {
                // Would implement Q8 decompression to F32
                Ok(data) // For now, return unchanged
            }
            _ => {
                // For other types or non-quantized, return unchanged
                Ok(data)
            }
        }
    }

    /// Get current position
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Reset position counter
    pub fn reset_position(&mut self) {
        self.position = 0;
    }

    /// Skip bytes in the stream
    pub fn skip_bytes(&mut self, count: usize) -> Result<()> {
        let mut buffer = vec![0u8; (count).min(8192)]; // Use reasonable buffer size
        let mut remaining = count;

        while remaining > 0 {
            let to_skip = remaining.min(buffer.len());
            self.reader.read_exact(&mut buffer[..to_skip])?;
            remaining -= to_skip;
            self.position += to_skip as u64;
        }

        Ok(())
    }
}

impl<R: Read + Seek> TensorReader<R> {
    /// Create a tensor reader with seeking support
    pub fn with_seek(reader: R) -> Self {
        Self { reader, position: 0 }
    }

    /// Seek to a specific position
    pub fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let new_pos = self.reader.seek(pos)?;
        self.position = new_pos;
        Ok(new_pos)
    }

    /// Read tensor data at a specific offset
    pub fn read_tensor_at_offset(
        &mut self,
        tensor_info: &TensorInfo,
        offset: u64,
        options: &TensorReadOptions,
    ) -> Result<TensorReadResult> {
        self.seek(SeekFrom::Start(offset))?;
        self.read_tensor_data_with_options(tensor_info, options)
    }

    /// Read multiple tensors by seeking to their offsets
    pub fn read_tensors_by_offset(
        &mut self,
        tensors: &[(u64, &TensorInfo)], // (offset, tensor_info) pairs
        options: &TensorReadOptions,
    ) -> Result<Vec<TensorReadResult>> {
        let mut results = Vec::with_capacity(tensors.len());

        for &(offset, tensor_info) in tensors {
            let result = self.read_tensor_at_offset(tensor_info, offset, options)?;
            results.push(result);
        }

        Ok(results)
    }
}

/// Utility functions for tensor reading
pub struct TensorReadUtils;

impl TensorReadUtils {
    /// Calculate optimal buffer size for reading a tensor
    pub fn optimal_buffer_size(tensor_info: &TensorInfo) -> usize {
        let tensor_size = tensor_info.expected_data_size() as usize;

        // Use powers of 2 for better memory alignment
        let base_size = match tensor_size {
            0..=4096 => 4096,             // 4KB for small tensors
            4097..=65_536 => 16_384,      // 16KB for medium tensors
            65_537..=1_048_576 => 65_536, // 64KB for large tensors
            _ => 262_144,                 // 256KB for very large tensors
        };

        base_size.min(tensor_size)
    }

    /// Check if a tensor should be read in chunks
    pub fn should_read_chunked(tensor_info: &TensorInfo, chunk_threshold: usize) -> bool {
        tensor_info.expected_data_size() as usize > chunk_threshold
    }

    /// Calculate memory requirements for reading tensors
    pub fn calculate_memory_requirements(tensor_infos: &[&TensorInfo]) -> TensorMemoryRequirements {
        let mut total_size = 0u64;
        let mut max_tensor_size = 0u64;
        let mut quantized_count = 0;
        let mut non_quantized_count = 0;

        for tensor_info in tensor_infos {
            let size = tensor_info.expected_data_size();
            total_size += size;
            max_tensor_size = max_tensor_size.max(size);

            if tensor_info.tensor_type().is_quantized() {
                quantized_count += 1;
            } else {
                non_quantized_count += 1;
            }
        }

        TensorMemoryRequirements {
            total_size: total_size as usize,
            max_tensor_size: max_tensor_size as usize,
            tensor_count: tensor_infos.len(),
            quantized_tensor_count: quantized_count,
            non_quantized_tensor_count: non_quantized_count,
            recommended_buffer_size: Self::optimal_buffer_size(
                tensor_infos.iter().max_by_key(|t| t.expected_data_size()).unwrap(),
            ),
        }
    }
}

/// Memory requirements for reading tensors
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMemoryRequirements {
    /// Total size of all tensors
    pub total_size: usize,
    /// Size of the largest tensor
    pub max_tensor_size: usize,
    /// Total number of tensors
    pub tensor_count: usize,
    /// Number of quantized tensors
    pub quantized_tensor_count: usize,
    /// Number of non-quantized tensors
    pub non_quantized_tensor_count: usize,
    /// Recommended buffer size for reading
    pub recommended_buffer_size: usize,
}

impl TensorMemoryRequirements {
    /// Get the average tensor size
    pub fn average_tensor_size(&self) -> usize {
        if self.tensor_count == 0 {
            0
        } else {
            self.total_size / self.tensor_count
        }
    }

    /// Check if memory requirements are reasonable
    pub fn is_reasonable(&self, available_memory: usize) -> bool {
        // Should use less than or equal to 80% of available memory
        self.total_size <= (available_memory * 4 / 5)
    }
}

impl std::fmt::Display for TensorReadResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorReadResult {{ bytes: {}, validated: {}, decompressed: {}{}}}",
            self.bytes_read,
            self.was_validated,
            self.was_decompressed,
            if let Some(checksum) = self.checksum {
                format!(", checksum: 0x{:08x}", checksum)
            } else {
                String::new()
            }
        )
    }
}

impl std::fmt::Display for TensorMemoryRequirements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorMemoryRequirements {{ total: {} bytes, max: {} bytes, count: {} ({} quantized, {} non-quantized), avg: {} bytes }}",
            self.total_size,
            self.max_tensor_size,
            self.tensor_count,
            self.quantized_tensor_count,
            self.non_quantized_tensor_count,
            self.average_tensor_size()
        )
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::tensor::TensorShape;
    use std::io::Cursor;

    fn create_test_tensor_info(name: &str, shape: Vec<u64>, tensor_type: TensorType) -> TensorInfo {
        let shape = TensorShape::new(shape).unwrap();
        TensorInfo::new(name.to_string(), shape, tensor_type, 0)
    }

    #[test]
    fn test_tensor_reader_creation() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let cursor = Cursor::new(data);
        let reader = TensorReader::new(cursor);

        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn test_read_tensor_data() {
        let data = vec![0u8; 16]; // 4 F32 values
        let cursor = Cursor::new(data);
        let mut reader = TensorReader::new(cursor);

        let tensor_info = create_test_tensor_info("test", vec![4], TensorType::F32);
        let options = TensorReadOptions {
            validate_alignment: false, // Skip alignment check for test data
            ..Default::default()
        };
        let result = reader.read_tensor_data_with_options(&tensor_info, &options).unwrap();

        assert_eq!(result.bytes_read, 16);
        assert_eq!(result.data.len(), 16);
        assert!(!result.was_decompressed);
    }

    #[test]
    fn test_read_tensor_with_options() {
        let data = vec![0u8; 8]; // 2 F32 values
        let cursor = Cursor::new(data);
        let mut reader = TensorReader::new(cursor);

        let tensor_info = create_test_tensor_info("test", vec![2], TensorType::F32);
        let options = TensorReadOptions {
            validate_integrity: true,
            validate_alignment: false, // Skip alignment check for test
            ..Default::default()
        };

        let result = reader.read_tensor_data_with_options(&tensor_info, &options).unwrap();
        assert!(result.was_validated);
        assert!(result.checksum.is_some());
    }

    #[test]
    fn test_read_multiple_tensors() {
        let data = vec![0u8; 24]; // Two tensors: 4 F32 + 2 F32
        let cursor = Cursor::new(data);
        let mut reader = TensorReader::new(cursor);

        let tensor1 = create_test_tensor_info("tensor1", vec![4], TensorType::F32);
        let tensor2 = create_test_tensor_info("tensor2", vec![2], TensorType::F32);
        let tensor_infos = vec![&tensor1, &tensor2];

        let options = TensorReadOptions {
            validate_alignment: false, // Skip alignment check for test data
            ..Default::default()
        };
        let results = reader.read_multiple_tensors(&tensor_infos, &options).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].bytes_read, 16);
        assert_eq!(results[1].bytes_read, 8);
    }

    #[test]
    fn test_read_tensor_chunked() {
        let data = vec![0u8; 16];
        let cursor = Cursor::new(data);
        let mut reader = TensorReader::new(cursor);

        let tensor_info = create_test_tensor_info("test", vec![4], TensorType::F32);
        let mut chunks = Vec::new();

        reader
            .read_tensor_chunked(&tensor_info, 8, |chunk| {
                chunks.push(chunk.to_vec());
                Ok(())
            })
            .unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), 8);
        assert_eq!(chunks[1].len(), 8);
    }

    #[test]
    fn test_size_limit() {
        let data = vec![0u8; 16];
        let cursor = Cursor::new(data);
        let mut reader = TensorReader::new(cursor);

        let tensor_info = create_test_tensor_info("test", vec![4], TensorType::F32);
        let options = TensorReadOptions {
            max_tensor_size: 8, // Smaller than tensor size (16)
            ..Default::default()
        };

        let result = reader.read_tensor_data_with_options(&tensor_info, &options);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_read_utils() {
        let small_tensor = create_test_tensor_info("small", vec![10], TensorType::F32);
        let large_tensor = create_test_tensor_info("large", vec![100000], TensorType::F32);

        let small_buffer = TensorReadUtils::optimal_buffer_size(&small_tensor);
        let large_buffer = TensorReadUtils::optimal_buffer_size(&large_tensor);

        assert!(small_buffer <= large_buffer);

        assert!(!TensorReadUtils::should_read_chunked(&small_tensor, 1000));
        assert!(TensorReadUtils::should_read_chunked(&large_tensor, 1000));
    }

    #[test]
    fn test_memory_requirements() {
        let tensor1 = create_test_tensor_info("t1", vec![100], TensorType::F32);
        let tensor2 = create_test_tensor_info("t2", vec![200], TensorType::Q4_0);
        let tensors = vec![&tensor1, &tensor2];

        let req = TensorReadUtils::calculate_memory_requirements(&tensors);
        assert_eq!(req.tensor_count, 2);
        assert_eq!(req.quantized_tensor_count, 1);
        assert_eq!(req.non_quantized_tensor_count, 1);
        assert!(req.total_size > 0);
        assert!(req.max_tensor_size >= req.average_tensor_size());
    }

    #[test]
    fn test_seeking_reader() {
        let data = vec![0u8; 32];
        let cursor = Cursor::new(data);
        let mut reader = TensorReader::with_seek(cursor);

        // Seek to position 16
        let pos = reader.seek(SeekFrom::Start(16)).unwrap();
        assert_eq!(pos, 16);
        assert_eq!(reader.position(), 16);
    }

    #[test]
    fn test_skip_bytes() {
        let data = vec![0u8; 32];
        let cursor = Cursor::new(data);
        let mut reader = TensorReader::new(cursor);

        reader.skip_bytes(10).unwrap();
        assert_eq!(reader.position(), 10);

        reader.skip_bytes(5).unwrap();
        assert_eq!(reader.position(), 15);
    }

    #[test]
    fn test_display_implementations() {
        let result = TensorReadResult {
            data: TensorData::new_owned(vec![1, 2, 3, 4]),
            bytes_read: 4,
            was_validated: true,
            was_decompressed: false,
            checksum: Some(0x12345678),
        };

        let display_str = format!("{}", result);
        assert!(display_str.contains("4"));
        assert!(display_str.contains("validated: true"));
        assert!(display_str.contains("0x12345678"));

        let req = TensorMemoryRequirements {
            total_size: 1000,
            max_tensor_size: 500,
            tensor_count: 2,
            quantized_tensor_count: 1,
            non_quantized_tensor_count: 1,
            recommended_buffer_size: 256,
        };

        let req_str = format!("{}", req);
        assert!(req_str.contains("1000"));
        assert!(req_str.contains("500"));
    }

    #[test]
    fn test_memory_requirements_reasonable() {
        let req = TensorMemoryRequirements {
            total_size: 800,
            max_tensor_size: 400,
            tensor_count: 2,
            quantized_tensor_count: 1,
            non_quantized_tensor_count: 1,
            recommended_buffer_size: 256,
        };

        assert!(req.is_reasonable(1000)); // 800 < 80% of 1000
        assert!(!req.is_reasonable(900)); // 800 >= 80% of 900
        assert_eq!(req.average_tensor_size(), 400); // 800 / 2
    }
}
