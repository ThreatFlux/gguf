//! File-based GGUF reader

#[cfg(feature = "std")]
use crate::error::{GGUFError, Result};
#[cfg(feature = "std")]
use crate::format::types::GGUFTensorType as TensorType;
#[cfg(feature = "std")]
use crate::format::{alignment::align_to_default, GGUFHeader, Metadata, TensorInfo};
#[cfg(feature = "std")]
use crate::tensor::{TensorData, TensorInfo as TensorInfoNew, TensorShape};
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::{BufReader, Read, Seek, SeekFrom};
#[cfg(feature = "std")]
use std::path::Path;

/// A reader for GGUF files
#[derive(Debug)]
pub struct GGUFFileReader<R> {
    /// The underlying reader
    reader: R,
    /// File header
    header: GGUFHeader,
    /// Metadata
    metadata: Metadata,
    /// Tensor information
    tensor_infos: Vec<TensorInfoNew>,
    /// Current position in the file
    position: u64,
    /// Start of tensor data section
    tensor_data_offset: u64,
}

/// Configuration for GGUF file reading
#[derive(Debug, Clone)]
pub struct GGUFReaderConfig {
    /// Whether to validate data integrity
    pub validate_integrity: bool,
    /// Whether to load tensor data immediately
    pub eager_load_tensors: bool,
    /// Maximum file size to read (0 = no limit)
    pub max_file_size: u64,
    /// Buffer size for reading
    pub buffer_size: usize,
    /// Whether to use memory mapping when available
    pub use_mmap: bool,
}

impl Default for GGUFReaderConfig {
    fn default() -> Self {
        Self {
            validate_integrity: true,
            eager_load_tensors: false,
            max_file_size: 0,
            buffer_size: 64 * 1024, // 64KB buffer
            use_mmap: false,
        }
    }
}

impl<R: Read + Seek> GGUFFileReader<R> {
    /// Create a new GGUF file reader with default configuration
    pub fn new(reader: R) -> Result<Self> {
        Self::with_config(reader, GGUFReaderConfig::default())
    }

    /// Create a new GGUF file reader with custom configuration
    pub fn with_config(mut reader: R, config: GGUFReaderConfig) -> Result<Self> {
        // Read and validate header
        let header = GGUFHeader::read_from(&mut reader)?;
        header.validate_comprehensive()?;

        // Check file size limits
        if config.max_file_size > 0 {
            let current_pos = reader.stream_position()?;
            let file_size = reader.seek(SeekFrom::End(0))?;
            reader.seek(SeekFrom::Start(current_pos))?;

            if file_size > config.max_file_size {
                return Err(GGUFError::Format(format!(
                    "File size {} exceeds maximum allowed size {}",
                    file_size, config.max_file_size
                )));
            }
        }

        // Read metadata
        let metadata = Metadata::read_from(&mut reader, header.metadata_kv_count)?;

        // Read tensor information
        let mut tensor_infos = Vec::with_capacity(header.tensor_count as usize);
        for _ in 0..header.tensor_count {
            let tensor_info = TensorInfo::read_from(&mut reader)?;

            // Convert to our TensorInfo format
            let shape = TensorShape::new(tensor_info.dimensions)?;
            let tensor_type = TensorType::from_u32(tensor_info.tensor_type)?;

            let new_tensor_info =
                TensorInfoNew::new(tensor_info.name, shape, tensor_type, tensor_info.offset);

            tensor_infos.push(new_tensor_info);
        }

        // Calculate tensor data section offset
        let current_position = reader.stream_position()?;
        let tensor_data_offset = align_to_default(current_position as usize) as u64;

        let mut gguf_reader = Self {
            reader,
            header,
            metadata,
            tensor_infos,
            position: current_position,
            tensor_data_offset,
        };

        // Eager load tensor data if requested
        if config.eager_load_tensors {
            gguf_reader.load_all_tensor_data()?;
        }

        // Validate integrity if requested
        if config.validate_integrity {
            gguf_reader.validate_integrity()?;
        }

        Ok(gguf_reader)
    }

    /// Get the file header
    pub fn header(&self) -> &GGUFHeader {
        &self.header
    }

    /// Get the metadata
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Get tensor information
    pub fn tensor_infos(&self) -> &[TensorInfoNew] {
        &self.tensor_infos
    }

    /// Get a specific tensor info by name
    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfoNew> {
        self.tensor_infos.iter().find(|t| t.name() == name)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_infos.iter().map(|t| t.name()).collect()
    }

    /// Get the number of tensors
    pub fn tensor_count(&self) -> usize {
        self.tensor_infos.len()
    }

    /// Load tensor data by name
    pub fn load_tensor_data(&mut self, name: &str) -> Result<Option<TensorData>> {
        // Find the tensor
        let tensor_index =
            self.tensor_infos.iter().position(|t| t.name() == name).ok_or_else(|| {
                GGUFError::InvalidTensorData(format!("Tensor '{}' not found", name))
            })?;

        let tensor_info = &self.tensor_infos[tensor_index];
        let data_size = tensor_info.expected_data_size() as usize;

        // Seek to tensor data
        let absolute_offset = self.tensor_data_offset + tensor_info.data_offset();
        self.reader.seek(SeekFrom::Start(absolute_offset))?;

        // Read tensor data
        let mut data = vec![0u8; data_size];
        self.reader.read_exact(&mut data)?;

        let tensor_data = TensorData::new_owned(data);

        // Store in tensor info (we need mutable access)
        // For now, return the data and let caller handle storage
        Ok(Some(tensor_data))
    }

    /// Load all tensor data
    pub fn load_all_tensor_data(&mut self) -> Result<()> {
        let tensor_names: Vec<String> =
            self.tensor_names().iter().map(|&s| s.to_string()).collect();

        for name in tensor_names {
            let data = self.load_tensor_data(&name)?;
            if let Some(tensor_data) = data {
                // Find the tensor again and set its data
                if let Some(tensor_info) = self.tensor_infos.iter_mut().find(|t| t.name() == name) {
                    tensor_info.set_data(tensor_data);
                }
            }
        }

        Ok(())
    }

    /// Read tensor data at a specific offset and size
    pub fn read_tensor_data_at(&mut self, offset: u64, size: usize) -> Result<TensorData> {
        let absolute_offset = self.tensor_data_offset + offset;
        self.reader.seek(SeekFrom::Start(absolute_offset))?;

        let mut data = vec![0u8; size];
        self.reader.read_exact(&mut data)?;

        Ok(TensorData::new_owned(data))
    }

    /// Get current position in file
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Get tensor data section offset
    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    /// Validate the integrity of the GGUF file
    pub fn validate_integrity(&mut self) -> Result<()> {
        // Check header consistency
        if self.header.tensor_count as usize != self.tensor_infos.len() {
            return Err(GGUFError::Format(
                "Header tensor count doesn't match actual tensor count".to_string(),
            ));
        }

        if self.header.metadata_kv_count as usize != self.metadata.len() {
            return Err(GGUFError::Format(
                "Header metadata count doesn't match actual metadata count".to_string(),
            ));
        }

        // Validate tensor infos
        for tensor_info in &self.tensor_infos {
            tensor_info.validate()?;
        }

        // Check for tensor offset overlaps
        let mut tensor_ranges: Vec<(u64, u64, &str)> = self
            .tensor_infos
            .iter()
            .map(|t| (t.data_offset(), t.expected_data_size(), t.name()))
            .collect();

        tensor_ranges.sort_by_key(|(offset, _, _)| *offset);

        for window in tensor_ranges.windows(2) {
            let (start_offset1, size1, name1) = window[0];
            let (start_offset2, _, name2) = window[1];

            if start_offset1 + size1 > start_offset2 {
                return Err(GGUFError::Format(format!(
                    "Tensor data overlap detected: '{}' ({}..{}) overlaps with '{}' ({}..)",
                    name1,
                    start_offset1,
                    start_offset1 + size1,
                    name2,
                    start_offset2
                )));
            }
        }

        Ok(())
    }

    /// Get a summary of the GGUF file
    pub fn summary(&self) -> GGUFFileSummary {
        let total_tensor_size: u64 = self.tensor_infos.iter().map(|t| t.expected_data_size()).sum();

        let loaded_tensor_count = self.tensor_infos.iter().filter(|t| t.has_data()).count();

        let tensor_types: HashMap<TensorType, usize> = {
            let mut types = HashMap::new();
            for tensor_info in &self.tensor_infos {
                *types.entry(tensor_info.tensor_type()).or_insert(0) += 1;
            }
            types
        };

        GGUFFileSummary {
            header: self.header.clone(),
            metadata_count: self.metadata.len(),
            tensor_count: self.tensor_infos.len(),
            loaded_tensor_count,
            total_tensor_size,
            tensor_data_offset: self.tensor_data_offset,
            tensor_types,
        }
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> GGUFMemoryUsage {
        let mut total_loaded_bytes = 0;
        let mut total_expected_bytes = 0;

        for tensor_info in &self.tensor_infos {
            total_expected_bytes += tensor_info.expected_data_size() as usize;
            if let Some(data) = tensor_info.data() {
                total_loaded_bytes += data.len();
            }
        }

        GGUFMemoryUsage {
            header_size: GGUFHeader::size(),
            metadata_size: self.metadata.serialized_size(),
            tensor_info_size: self
                .tensor_infos
                .iter()
                .map(|t| t.name().len() + 32) // Approximate
                .sum(),
            total_expected_tensor_bytes: total_expected_bytes,
            total_loaded_tensor_bytes: total_loaded_bytes,
        }
    }

    /// Seek to a specific position in the file
    pub fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let new_pos = self.reader.seek(pos)?;
        self.position = new_pos;
        Ok(new_pos)
    }

    /// Check if all tensor data is loaded
    pub fn is_fully_loaded(&self) -> bool {
        self.tensor_infos.iter().all(|t| t.has_data())
    }

    /// Unload all tensor data to save memory
    pub fn unload_all_tensor_data(&mut self) {
        for tensor_info in &mut self.tensor_infos {
            tensor_info.clear_data();
        }
    }

    /// Load only specific tensors by name patterns
    pub fn load_tensors_matching<F>(&mut self, predicate: F) -> Result<usize>
    where
        F: Fn(&str) -> bool,
    {
        let mut loaded_count = 0;
        let tensor_names: Vec<String> = self
            .tensor_infos
            .iter()
            .filter(|t| predicate(t.name()))
            .map(|t| t.name().to_string())
            .collect();

        for name in tensor_names {
            if let Some(data) = self.load_tensor_data(&name)? {
                if let Some(tensor_info) = self.tensor_infos.iter_mut().find(|t| t.name() == name) {
                    tensor_info.set_data(data);
                    loaded_count += 1;
                }
            }
        }

        Ok(loaded_count)
    }

    /// Get underlying reader (consuming the GGUFFileReader)
    pub fn into_inner(self) -> R {
        self.reader
    }
}

/// Summary information about a GGUF file
#[derive(Debug, Clone)]
pub struct GGUFFileSummary {
    /// File header
    pub header: GGUFHeader,
    /// Number of metadata entries
    pub metadata_count: usize,
    /// Total number of tensors
    pub tensor_count: usize,
    /// Number of loaded tensors
    pub loaded_tensor_count: usize,
    /// Total size of all tensor data
    pub total_tensor_size: u64,
    /// Offset where tensor data begins
    pub tensor_data_offset: u64,
    /// Count of each tensor type
    pub tensor_types: HashMap<TensorType, usize>,
}

/// Memory usage information for a GGUF file
#[derive(Debug, Clone)]
pub struct GGUFMemoryUsage {
    /// Size of the header
    pub header_size: usize,
    /// Size of the metadata section
    pub metadata_size: usize,
    /// Size of tensor info section
    pub tensor_info_size: usize,
    /// Expected total tensor data size
    pub total_expected_tensor_bytes: usize,
    /// Actually loaded tensor data size
    pub total_loaded_tensor_bytes: usize,
}

impl GGUFMemoryUsage {
    /// Get total overhead (non-tensor data)
    pub fn overhead_bytes(&self) -> usize {
        self.header_size + self.metadata_size + self.tensor_info_size
    }

    /// Get total size including loaded tensor data
    pub fn total_loaded_bytes(&self) -> usize {
        self.overhead_bytes() + self.total_loaded_tensor_bytes
    }

    /// Get compression ratio (loaded / expected)
    pub fn compression_ratio(&self) -> f32 {
        if self.total_expected_tensor_bytes == 0 {
            0.0
        } else {
            self.total_loaded_tensor_bytes as f32 / self.total_expected_tensor_bytes as f32
        }
    }
}

/// Convenience function to open a GGUF file from a path
pub fn open_gguf_file<P: AsRef<Path>>(path: P) -> Result<GGUFFileReader<BufReader<File>>> {
    let file = File::open(path)?;
    let buf_reader = BufReader::new(file);
    GGUFFileReader::new(buf_reader)
}

/// Convenience function to open a GGUF file with custom configuration
pub fn open_gguf_file_with_config<P: AsRef<Path>>(
    path: P,
    config: GGUFReaderConfig,
) -> Result<GGUFFileReader<BufReader<File>>> {
    let file = File::open(path)?;
    let buf_reader = BufReader::new(file);
    GGUFFileReader::with_config(buf_reader, config)
}

impl std::fmt::Display for GGUFFileSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GGUF File Summary:")?;
        writeln!(f, "  Version: {}", self.header.version)?;
        writeln!(f, "  Tensors: {} ({} loaded)", self.tensor_count, self.loaded_tensor_count)?;
        writeln!(f, "  Metadata entries: {}", self.metadata_count)?;
        writeln!(f, "  Total tensor size: {} bytes", self.total_tensor_size)?;
        writeln!(f, "  Tensor data offset: {}", self.tensor_data_offset)?;
        writeln!(f, "  Tensor types:")?;

        for (tensor_type, count) in &self.tensor_types {
            writeln!(f, "    {}: {}", tensor_type.name(), count)?;
        }

        Ok(())
    }
}

impl std::fmt::Display for GGUFMemoryUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GGUF Memory Usage:")?;
        writeln!(f, "  Header: {} bytes", self.header_size)?;
        writeln!(f, "  Metadata: {} bytes", self.metadata_size)?;
        writeln!(f, "  Tensor info: {} bytes", self.tensor_info_size)?;
        writeln!(f, "  Overhead: {} bytes", self.overhead_bytes())?;
        writeln!(f, "  Expected tensor data: {} bytes", self.total_expected_tensor_bytes)?;
        writeln!(f, "  Loaded tensor data: {} bytes", self.total_loaded_tensor_bytes)?;
        writeln!(f, "  Total loaded: {} bytes", self.total_loaded_bytes())?;
        writeln!(f, "  Compression ratio: {:.2}%", self.compression_ratio() * 100.0)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::constants::*;
    use std::io::Cursor;

    fn create_minimal_gguf_data() -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata entry

        // Metadata
        data.extend_from_slice(&4u64.to_le_bytes()); // key length
        data.extend_from_slice(b"test"); // key
        data.extend_from_slice(&8u32.to_le_bytes()); // string type
        data.extend_from_slice(&5u64.to_le_bytes()); // value length
        data.extend_from_slice(b"value"); // value

        // Tensor info
        data.extend_from_slice(&11u64.to_le_bytes()); // name length
        data.extend_from_slice(b"test_tensor"); // name
        data.extend_from_slice(&2u32.to_le_bytes()); // 2 dimensions
        data.extend_from_slice(&2u64.to_le_bytes()); // dim 0
        data.extend_from_slice(&3u64.to_le_bytes()); // dim 1
        data.extend_from_slice(&0u32.to_le_bytes()); // F32 type
        data.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Align to 32 bytes for tensor data
        while data.len() % 32 != 0 {
            data.push(0);
        }

        // Tensor data (2x3 F32 = 24 bytes)
        data.extend_from_slice(&[0u8; 24]);

        data
    }

    #[test]
    fn test_gguf_file_reader_creation() {
        let data = create_minimal_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFFileReader::new(cursor).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        assert_eq!(reader.metadata().len(), 1);
        assert_eq!(reader.header().tensor_count, 1);
    }

    #[test]
    fn test_gguf_reader_config() {
        let data = create_minimal_gguf_data();
        let cursor = Cursor::new(data);

        let config = GGUFReaderConfig {
            validate_integrity: true,
            eager_load_tensors: false,
            max_file_size: 1024,
            buffer_size: 8192,
            use_mmap: false,
        };

        let reader = GGUFFileReader::with_config(cursor, config).unwrap();
        assert!(!reader.is_fully_loaded());
    }

    #[test]
    fn test_tensor_operations() {
        let data = create_minimal_gguf_data();
        let cursor = Cursor::new(data);

        let mut reader = GGUFFileReader::new(cursor).unwrap();

        // Test tensor lookup
        let tensor_names = reader.tensor_names();
        assert_eq!(tensor_names.len(), 1);
        assert_eq!(tensor_names[0], "test_tensor");

        let tensor_info = reader.get_tensor_info("test_tensor").unwrap();
        assert_eq!(tensor_info.name(), "test_tensor");
        assert_eq!(tensor_info.element_count(), 6); // 2x3

        // Test data loading
        let tensor_data = reader.load_tensor_data("test_tensor").unwrap();
        assert!(tensor_data.is_some());
        assert_eq!(tensor_data.unwrap().len(), 24); // 6 F32 = 24 bytes
    }

    #[test]
    fn test_file_summary() {
        let data = create_minimal_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFFileReader::new(cursor).unwrap();
        let summary = reader.summary();

        assert_eq!(summary.tensor_count, 1);
        assert_eq!(summary.metadata_count, 1);
        assert_eq!(summary.loaded_tensor_count, 0);
        assert_eq!(summary.total_tensor_size, 24);
    }

    #[test]
    fn test_memory_usage() {
        let data = create_minimal_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFFileReader::new(cursor).unwrap();
        let memory_usage = reader.memory_usage();

        assert_eq!(memory_usage.header_size, 24);
        assert!(memory_usage.metadata_size > 0);
        assert_eq!(memory_usage.total_expected_tensor_bytes, 24);
        assert_eq!(memory_usage.total_loaded_tensor_bytes, 0);
        assert_eq!(memory_usage.compression_ratio(), 0.0);
    }

    #[test]
    fn test_integrity_validation() {
        let data = create_minimal_gguf_data();
        let cursor = Cursor::new(data);

        let mut reader = GGUFFileReader::new(cursor).unwrap();
        assert!(reader.validate_integrity().is_ok());
    }

    #[test]
    fn test_eager_loading() {
        let data = create_minimal_gguf_data();
        let cursor = Cursor::new(data);

        let config = GGUFReaderConfig { eager_load_tensors: true, ..Default::default() };

        let reader = GGUFFileReader::with_config(cursor, config).unwrap();
        assert!(reader.is_fully_loaded());
    }

    #[test]
    fn test_selective_loading() {
        let data = create_minimal_gguf_data();
        let cursor = Cursor::new(data);

        let mut reader = GGUFFileReader::new(cursor).unwrap();

        let loaded_count = reader.load_tensors_matching(|name| name.contains("test")).unwrap();
        assert_eq!(loaded_count, 1);
    }

    #[test]
    fn test_display_implementations() {
        let data = create_minimal_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFFileReader::new(cursor).unwrap();

        let summary = reader.summary();
        let summary_str = format!("{}", summary);
        assert!(summary_str.contains("GGUF File Summary"));
        assert!(summary_str.contains("F32"));

        let memory_usage = reader.memory_usage();
        let memory_str = format!("{}", memory_usage);
        assert!(memory_str.contains("Memory Usage"));
        assert!(memory_str.contains("bytes"));
    }

    #[test]
    fn test_file_size_limit() {
        let data = create_minimal_gguf_data();
        let cursor = Cursor::new(data.clone());

        let config = GGUFReaderConfig {
            max_file_size: (data.len() - 1) as u64, // Set limit below actual size
            ..Default::default()
        };

        let result = GGUFFileReader::with_config(cursor, config);
        assert!(result.is_err());
    }
}
