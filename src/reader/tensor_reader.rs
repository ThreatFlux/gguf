//! Tensor-specific reading utilities

use crate::error::{GGUFError, Result};
use crate::tensor::{TensorData, TensorInfo};
use std::io::{Read, Seek, SeekFrom};

const MAX_READ_CHUNK_SIZE: usize = 1024 * 1024;

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
    /// Whether to validate the descriptor offset against GGUF's minimum 8-byte alignment
    pub validate_alignment: bool,
    /// Whether to compute the crate's non-cryptographic payload checksum.
    ///
    /// GGUF carries no reference checksum to verify, so this records a value
    /// for caller comparison; it does not establish integrity or authenticity.
    pub compute_checksum: bool,
    /// Whether to decompress quantized data
    pub decompress_quantized: bool,
    /// Maximum tensor size to read (0 = no limit)
    pub max_tensor_size: usize,
    /// Preferred buffer size for reading; temporary chunks are capped at 1 MiB.
    pub buffer_size: usize,
}

impl Default for TensorReadOptions {
    fn default() -> Self {
        Self {
            validate_alignment: true,
            compute_checksum: false,
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
        let expected_size =
            usize::try_from(tensor_info.checked_expected_data_size()?).map_err(|_| {
                GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' size does not fit this platform",
                    tensor_info.name()
                ))
            })?;

        // Check size limits
        if options.max_tensor_size > 0 && expected_size > options.max_tensor_size {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' size {} exceeds maximum {}",
                tensor_info.name(),
                expected_size,
                options.max_tensor_size
            )));
        }

        // Reject descriptor/configuration errors before consuming input or
        // allocating a payload-sized buffer.
        if options.validate_alignment {
            self.validate_tensor_alignment(tensor_info)?;
        }
        if options.decompress_quantized && tensor_info.tensor_type().is_quantized() {
            return Err(GGUFError::FeatureUnavailable(format!(
                "decompression for {} tensors",
                tensor_info.tensor_type()
            )));
        }

        // Grow only after each bounded read so a truncated hostile input cannot force a
        // descriptor-sized allocation before any tensor bytes have arrived.
        let data = self.read_exact_owned_tracking(expected_size, options.buffer_size)?;

        let tensor_data = TensorData::new_owned(data);
        let checksum = options.compute_checksum.then(|| tensor_data.checksum());

        Ok(TensorReadResult {
            data: tensor_data,
            bytes_read: expected_size,
            was_decompressed: false,
            checksum,
        })
    }

    /// Read multiple tensors efficiently
    pub fn read_multiple_tensors(
        &mut self,
        tensor_infos: &[&TensorInfo],
        options: &TensorReadOptions,
    ) -> Result<Vec<TensorReadResult>> {
        let mut results = Vec::new();
        results.try_reserve_exact(tensor_infos.len()).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor read results".to_string())
        })?;

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
        let total_size =
            usize::try_from(tensor_info.checked_expected_data_size()?).map_err(|_| {
                GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' size does not fit this platform",
                    tensor_info.name()
                ))
            })?;
        if total_size > 0 && chunk_size == 0 {
            return Err(GGUFError::InvalidTensorData(
                "Tensor read chunk size must be greater than zero".to_string(),
            ));
        }
        let mut remaining = total_size;
        let buffer_size = chunk_size.min(remaining).min(MAX_READ_CHUNK_SIZE);
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(buffer_size).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor read chunk".to_string())
        })?;
        buffer.resize(buffer_size, 0);

        while remaining > 0 {
            let to_read = buffer.len().min(remaining);
            self.read_exact_tracking(&mut buffer[..to_read])?;

            callback(&buffer[..to_read])?;
            remaining -= to_read;
        }

        Ok(())
    }

    /// Validate tensor data alignment
    fn validate_tensor_alignment(&self, tensor_info: &TensorInfo) -> Result<()> {
        // TensorReader does not have access to `general.alignment`; the full file and
        // stream readers enforce that exact value. Every valid declared alignment is
        // a multiple of eight, so this is the strongest deterministic check available.
        const MIN_GGUF_ALIGNMENT: u64 = 8;
        if !tensor_info.data_offset().is_multiple_of(MIN_GGUF_ALIGNMENT) {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' offset is not aligned to at least {} bytes",
                tensor_info.name(),
                MIN_GGUF_ALIGNMENT
            )));
        }

        Ok(())
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
        let mut buffer = [0u8; 8192];
        let mut remaining = count;

        while remaining > 0 {
            let to_skip = remaining.min(buffer.len());
            self.read_exact_tracking(&mut buffer[..to_skip])?;
            remaining -= to_skip;
        }

        Ok(())
    }

    fn read_exact_owned_tracking(&mut self, size: usize, buffer_size: usize) -> Result<Vec<u8>> {
        if size == 0 {
            return Ok(Vec::new());
        }
        if buffer_size == 0 {
            return Err(GGUFError::InvalidTensorData(
                "Tensor read buffer size must be greater than zero".to_string(),
            ));
        }

        let chunk_size = buffer_size.min(size).min(MAX_READ_CHUNK_SIZE);
        let mut chunk = Vec::new();
        chunk.try_reserve_exact(chunk_size).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor read chunk".to_string())
        })?;
        chunk.resize(chunk_size, 0);
        let mut data = Vec::new();
        let mut remaining = size;
        while remaining > 0 {
            let to_read = remaining.min(chunk.len());
            self.read_exact_tracking(&mut chunk[..to_read])?;
            data.try_reserve(to_read).map_err(|_| {
                GGUFError::InvalidTensorData("Unable to allocate tensor data buffer".to_string())
            })?;
            data.extend_from_slice(&chunk[..to_read]);
            remaining -= to_read;
        }
        Ok(data)
    }

    fn read_exact_tracking(&mut self, mut buffer: &mut [u8]) -> Result<()> {
        while !buffer.is_empty() {
            match self.reader.read(buffer) {
                Ok(0) => return Err(GGUFError::UnexpectedEof),
                Ok(bytes_read) => {
                    self.advance(bytes_read)?;
                    let (_, remaining) = buffer.split_at_mut(bytes_read);
                    buffer = remaining;
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    fn advance(&mut self, bytes: usize) -> Result<()> {
        let bytes = u64::try_from(bytes)
            .map_err(|_| GGUFError::InvalidTensorData("Read size does not fit u64".to_string()))?;
        self.position = self.position.checked_add(bytes).ok_or_else(|| {
            GGUFError::InvalidTensorData("Tensor reader position overflows u64".to_string())
        })?;
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
        let mut results = Vec::new();
        results.try_reserve_exact(tensors.len()).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor read results".to_string())
        })?;

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
        let tensor_size = tensor_info
            .checked_expected_data_size()
            .ok()
            .and_then(|size| usize::try_from(size).ok())
            .unwrap_or(usize::MAX);

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
        tensor_info
            .checked_expected_data_size()
            .ok()
            .and_then(|size| usize::try_from(size).ok())
            .is_none_or(|size| size > chunk_threshold)
    }

    /// Calculate memory requirements for reading tensors
    pub fn calculate_memory_requirements(tensor_infos: &[&TensorInfo]) -> TensorMemoryRequirements {
        let mut total_size = 0u64;
        let mut max_tensor_size = 0u64;
        let mut quantized_count = 0;
        let mut non_quantized_count = 0;

        for tensor_info in tensor_infos {
            let size = tensor_info.expected_data_size();
            total_size = total_size.saturating_add(size);
            max_tensor_size = max_tensor_size.max(size);

            if tensor_info.tensor_type().is_quantized() {
                quantized_count += 1;
            } else {
                non_quantized_count += 1;
            }
        }

        TensorMemoryRequirements {
            total_size: usize::try_from(total_size).unwrap_or(usize::MAX),
            max_tensor_size: usize::try_from(max_tensor_size).unwrap_or(usize::MAX),
            tensor_count: tensor_infos.len(),
            quantized_tensor_count: quantized_count,
            non_quantized_tensor_count: non_quantized_count,
            recommended_buffer_size: tensor_infos
                .iter()
                .max_by_key(|tensor| tensor.expected_data_size())
                .map_or(0, |tensor| Self::optimal_buffer_size(tensor)),
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
        self.total_size.checked_div(self.tensor_count).unwrap_or(0)
    }

    /// Check if memory requirements are reasonable
    pub fn is_reasonable(&self, available_memory: usize) -> bool {
        // Should use less than or equal to 80% of available memory
        self.total_size <= available_memory.saturating_mul(4) / 5
    }
}

impl std::fmt::Display for TensorReadResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorReadResult {{ bytes: {}, decompressed: {}{}}}",
            self.bytes_read,
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
    use crate::tensor::{TensorShape, TensorType};
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
            compute_checksum: true,
            validate_alignment: false, // Skip alignment check for test
            ..Default::default()
        };

        let result = reader.read_tensor_data_with_options(&tensor_info, &options).unwrap();
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
    fn test_zero_sized_read_buffers_are_rejected() {
        let tensor_info = create_test_tensor_info("test", vec![1], TensorType::F32);
        let mut reader = TensorReader::new(Cursor::new(vec![0u8; 4]));
        let options =
            TensorReadOptions { validate_alignment: false, buffer_size: 0, ..Default::default() };
        assert!(reader.read_tensor_data_with_options(&tensor_info, &options).is_err());

        let mut reader = TensorReader::new(Cursor::new(vec![0u8; 4]));
        assert!(reader.read_tensor_chunked(&tensor_info, 0, |_| Ok(())).is_err());
    }

    #[test]
    fn test_unimplemented_decompression_is_reported() {
        let tensor_info = create_test_tensor_info("test", vec![32], TensorType::Q4_0);
        let mut reader = TensorReader::new(Cursor::new(vec![0u8; 18]));
        let options = TensorReadOptions {
            validate_alignment: false,
            decompress_quantized: true,
            ..Default::default()
        };
        assert!(matches!(
            reader.read_tensor_data_with_options(&tensor_info, &options),
            Err(GGUFError::FeatureUnavailable(_))
        ));
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn invalid_alignment_is_rejected_before_reading() {
        let mut tensor_info = create_test_tensor_info("test", vec![1], TensorType::F32);
        tensor_info.data_offset = 1;
        let mut reader = TensorReader::new(Cursor::new(vec![0u8; 4]));

        assert!(reader.read_tensor_data(&tensor_info).is_err());
        assert_eq!(reader.position(), 0);
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
            was_decompressed: false,
            checksum: Some(0x12345678),
        };

        let display_str = format!("{}", result);
        assert!(display_str.contains("4"));
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
