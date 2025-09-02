//! Stream-based GGUF reader for non-seekable streams

use crate::error::{GGUFError, Result};
use crate::format::types::GGUFTensorType as TensorType;
use crate::format::{alignment::calculate_default_padding, GGUFHeader, Metadata, TensorInfo};
use crate::tensor::{TensorData, TensorInfo as TensorInfoNew, TensorShape};
use std::collections::HashMap;
use std::io::{BufReader, Read};

/// A reader for GGUF files from non-seekable streams
#[derive(Debug)]
pub struct GGUFStreamReader<R> {
    /// The underlying reader
    reader: R,
    /// File header
    header: GGUFHeader,
    /// Metadata
    metadata: Metadata,
    /// Tensor information
    tensor_infos: Vec<TensorInfoNew>,
    /// Current position in the stream
    position: u64,
    /// Whether we've reached the tensor data section
    at_tensor_data: bool,
}

/// Configuration for stream reading
#[derive(Debug, Clone)]
pub struct StreamReaderConfig {
    /// Buffer size for internal buffering
    pub buffer_size: usize,
    /// Whether to validate checksums
    pub validate_checksums: bool,
    /// Maximum metadata size to prevent DoS
    pub max_metadata_size: usize,
    /// Maximum number of tensors to prevent DoS
    pub max_tensor_count: usize,
}

impl Default for StreamReaderConfig {
    fn default() -> Self {
        Self {
            buffer_size: 64 * 1024,              // 64KB
            validate_checksums: false,           // Expensive for streams
            max_metadata_size: 16 * 1024 * 1024, // 16MB
            max_tensor_count: 100_000,
        }
    }
}

impl<R: Read> GGUFStreamReader<R> {
    /// Create a new stream reader with default configuration
    pub fn new(reader: R) -> Result<Self> {
        Self::with_config(reader, StreamReaderConfig::default())
    }

    /// Create a new stream reader with custom configuration
    pub fn with_config(mut reader: R, config: StreamReaderConfig) -> Result<Self> {
        let mut position = 0u64;

        // Read header
        let header = GGUFHeader::read_from(&mut reader)?;
        header.validate_comprehensive()?;
        position += GGUFHeader::size() as u64;

        // Check limits
        if header.tensor_count > config.max_tensor_count as u64 {
            return Err(GGUFError::Format(format!(
                "Too many tensors: {} exceeds limit of {}",
                header.tensor_count, config.max_tensor_count
            )));
        }

        // Read metadata
        let metadata = Metadata::read_from(&mut reader, header.metadata_kv_count)?;
        if metadata.serialized_size() > config.max_metadata_size {
            return Err(GGUFError::Format(format!(
                "Metadata too large: {} bytes exceeds limit of {}",
                metadata.serialized_size(),
                config.max_metadata_size
            )));
        }
        position += metadata.serialized_size() as u64;

        // Read tensor information
        let mut tensor_infos = Vec::with_capacity(header.tensor_count as usize);
        for _ in 0..header.tensor_count {
            let tensor_info = TensorInfo::read_from(&mut reader)?;
            position += tensor_info.serialized_size() as u64;

            // Convert to our TensorInfo format
            let shape = TensorShape::new(tensor_info.dimensions)?;
            let tensor_type = TensorType::from_u32(tensor_info.tensor_type)?;

            let new_tensor_info =
                TensorInfoNew::new(tensor_info.name, shape, tensor_type, tensor_info.offset);

            tensor_infos.push(new_tensor_info);
        }

        // Calculate alignment padding
        let padding_size = calculate_default_padding(position as usize);
        if padding_size > 0 {
            let mut padding = vec![0u8; padding_size];
            reader.read_exact(&mut padding)?;
            position += padding_size as u64;
        }

        Ok(Self { reader, header, metadata, tensor_infos, position, at_tensor_data: true })
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

    /// Get tensor names in order
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_infos.iter().map(|t| t.name()).collect()
    }

    /// Get the number of tensors
    pub fn tensor_count(&self) -> usize {
        self.tensor_infos.len()
    }

    /// Read the next tensor's data in stream order
    pub fn read_next_tensor(&mut self) -> Result<Option<(String, TensorData)>> {
        if !self.at_tensor_data {
            return Ok(None);
        }

        // Find the tensor with the smallest offset that we haven't read yet
        let next_tensor = self
            .tensor_infos
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.has_data())
            .min_by_key(|(_, t)| t.data_offset());

        if let Some((index, tensor_info)) = next_tensor {
            let data_size = tensor_info.expected_data_size() as usize;
            let target_position = tensor_info.data_offset();
            let tensor_name = tensor_info.name().to_string();
            let tensor_shape = tensor_info.shape().clone();
            let tensor_type = tensor_info.tensor_type();

            // Skip to the right position if needed
            if self.position < target_position {
                let skip_bytes = (target_position - self.position) as usize;
                self.skip_bytes(skip_bytes)?;
            }

            // Read tensor data
            let mut data = vec![0u8; data_size];
            self.reader.read_exact(&mut data)?;
            self.position += data_size as u64;

            let tensor_data = TensorData::new_owned(data);

            // Update the tensor info to mark it as having data
            if let Some(tensor_info_mut) = self.tensor_infos.get_mut(index) {
                *tensor_info_mut = TensorInfoNew::with_data(
                    tensor_name.clone(),
                    tensor_shape,
                    tensor_type,
                    target_position,
                    tensor_data.clone(),
                );
            }

            Ok(Some((tensor_name, tensor_data)))
        } else {
            Ok(None)
        }
    }

    /// Read all tensors in stream order
    pub fn read_all_tensors(&mut self) -> Result<HashMap<String, TensorData>> {
        let mut tensors = HashMap::new();

        while let Some((name, data)) = self.read_next_tensor()? {
            tensors.insert(name, data);
        }

        Ok(tensors)
    }

    /// Skip a certain number of bytes in the stream
    fn skip_bytes(&mut self, count: usize) -> Result<()> {
        const SKIP_BUFFER_SIZE: usize = 8192;
        let mut buffer = vec![0u8; SKIP_BUFFER_SIZE.min(count)];
        let mut remaining = count;

        while remaining > 0 {
            let to_read = remaining.min(SKIP_BUFFER_SIZE);
            self.reader.read_exact(&mut buffer[..to_read])?;
            remaining -= to_read;
            self.position += to_read as u64;
        }

        Ok(())
    }

    /// Get current position in the stream
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Check if we're at the tensor data section
    pub fn at_tensor_data(&self) -> bool {
        self.at_tensor_data
    }

    /// Create a streaming iterator over tensors
    pub fn tensor_iterator(self) -> TensorIterator<R> {
        TensorIterator::new(self)
    }

    /// Get a summary of what we've read so far
    pub fn summary(&self) -> StreamReaderSummary {
        let tensor_types: HashMap<TensorType, usize> = {
            let mut types = HashMap::new();
            for tensor_info in &self.tensor_infos {
                *types.entry(tensor_info.tensor_type()).or_insert(0) += 1;
            }
            types
        };

        let total_tensor_size: u64 = self.tensor_infos.iter().map(|t| t.expected_data_size()).sum();

        StreamReaderSummary {
            header: self.header.clone(),
            metadata_count: self.metadata.len(),
            tensor_count: self.tensor_infos.len(),
            total_tensor_size,
            current_position: self.position,
            tensor_types,
        }
    }

    /// Validate the stream data we've read so far
    pub fn validate(&self) -> Result<()> {
        // Validate header consistency
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

        Ok(())
    }

    /// Convert to underlying reader (consuming the stream reader)
    pub fn into_inner(self) -> R {
        self.reader
    }
}

/// Iterator over tensors in a stream
pub struct TensorIterator<R> {
    reader: GGUFStreamReader<R>,
    finished: bool,
}

impl<R: Read> TensorIterator<R> {
    fn new(reader: GGUFStreamReader<R>) -> Self {
        Self { reader, finished: false }
    }
}

impl<R: Read> Iterator for TensorIterator<R> {
    type Item = Result<(String, TensorData)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        match self.reader.read_next_tensor() {
            Ok(Some(tensor)) => Some(Ok(tensor)),
            Ok(None) => {
                self.finished = true;
                None
            }
            Err(e) => {
                self.finished = true;
                Some(Err(e))
            }
        }
    }
}

/// Summary of stream reading progress
#[derive(Debug, Clone)]
pub struct StreamReaderSummary {
    /// File header
    pub header: GGUFHeader,
    /// Number of metadata entries
    pub metadata_count: usize,
    /// Total number of tensors
    pub tensor_count: usize,
    /// Total size of all tensor data
    pub total_tensor_size: u64,
    /// Current position in stream
    pub current_position: u64,
    /// Count of each tensor type
    pub tensor_types: HashMap<TensorType, usize>,
}

/// Convenience function to create a stream reader from any Read type
pub fn stream_reader_from_read<R: Read>(reader: R) -> Result<GGUFStreamReader<R>> {
    GGUFStreamReader::new(reader)
}

/// Convenience function to create a buffered stream reader
pub fn buffered_stream_reader<R: Read>(reader: R) -> Result<GGUFStreamReader<BufReader<R>>> {
    let buf_reader = BufReader::new(reader);
    GGUFStreamReader::new(buf_reader)
}

impl std::fmt::Display for StreamReaderSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GGUF Stream Summary:")?;
        writeln!(f, "  Version: {}", self.header.version)?;
        writeln!(f, "  Tensors: {}", self.tensor_count)?;
        writeln!(f, "  Metadata entries: {}", self.metadata_count)?;
        writeln!(f, "  Total tensor size: {} bytes", self.total_tensor_size)?;
        writeln!(f, "  Current position: {} bytes", self.current_position)?;
        writeln!(
            f,
            "  Progress: {:.1}%",
            if self.total_tensor_size > 0 {
                (self.current_position as f64
                    / (self.current_position + self.total_tensor_size) as f64)
                    * 100.0
            } else {
                100.0
            }
        )?;
        writeln!(f, "  Tensor types:")?;

        for (tensor_type, count) in &self.tensor_types {
            writeln!(f, "    {}: {}", tensor_type.name(), count)?;
        }

        Ok(())
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::format::constants::*;
    use std::io::Cursor;

    fn create_stream_gguf_data() -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&2u64.to_le_bytes()); // 2 tensors
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata entry

        // Metadata
        data.extend_from_slice(&4u64.to_le_bytes()); // key length
        data.extend_from_slice(b"name"); // key
        data.extend_from_slice(&8u32.to_le_bytes()); // string type
        data.extend_from_slice(&5u64.to_le_bytes()); // value length
        data.extend_from_slice(b"model"); // value

        // Store positions where we'll write offsets
        let mut offset_positions = Vec::new();

        // Tensor info 1
        data.extend_from_slice(&8u64.to_le_bytes()); // name length
        data.extend_from_slice(b"tensor_a"); // name
        data.extend_from_slice(&1u32.to_le_bytes()); // 1 dimension
        data.extend_from_slice(&4u64.to_le_bytes()); // dim 0
        data.extend_from_slice(&0u32.to_le_bytes()); // F32 type
        offset_positions.push(data.len()); // Remember where offset goes
        data.extend_from_slice(&0u64.to_le_bytes()); // offset placeholder

        // Tensor info 2
        data.extend_from_slice(&8u64.to_le_bytes()); // name length
        data.extend_from_slice(b"tensor_b"); // name
        data.extend_from_slice(&1u32.to_le_bytes()); // 1 dimension
        data.extend_from_slice(&3u64.to_le_bytes()); // dim 0
        data.extend_from_slice(&0u32.to_le_bytes()); // F32 type
        offset_positions.push(data.len()); // Remember where offset goes
        data.extend_from_slice(&0u64.to_le_bytes()); // offset placeholder

        // Align to 32 bytes
        while data.len() % 32 != 0 {
            data.push(0);
        }

        // Now we know where tensor data starts - update the offsets
        let tensor_data_start = data.len() as u64;

        // Update tensor A offset
        let tensor_a_pos = offset_positions[0];
        data[tensor_a_pos..tensor_a_pos + 8].copy_from_slice(&tensor_data_start.to_le_bytes());

        // Update tensor B offset (after tensor A)
        let tensor_b_pos = offset_positions[1];
        let tensor_b_offset = tensor_data_start + 16; // After tensor A (16 bytes)
        data[tensor_b_pos..tensor_b_pos + 8].copy_from_slice(&tensor_b_offset.to_le_bytes());

        // Tensor data A (4 F32 = 16 bytes)
        data.extend_from_slice(&[0u8; 16]);

        // Tensor data B (3 F32 = 12 bytes)
        data.extend_from_slice(&[0u8; 12]);

        data
    }

    #[test]
    fn test_stream_reader_creation() {
        let data = create_stream_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFStreamReader::new(cursor).unwrap();
        assert_eq!(reader.tensor_count(), 2);
        assert_eq!(reader.metadata().len(), 1);
        assert!(reader.at_tensor_data());
    }

    #[test]
    fn test_stream_reader_config() {
        let data = create_stream_gguf_data();
        let cursor = Cursor::new(data);

        let config = StreamReaderConfig {
            buffer_size: 1024,
            validate_checksums: true,
            max_metadata_size: 1024,
            max_tensor_count: 10,
        };

        let reader = GGUFStreamReader::with_config(cursor, config).unwrap();
        assert_eq!(reader.tensor_count(), 2);
    }

    #[test]
    fn test_read_next_tensor() {
        let data = create_stream_gguf_data();
        let cursor = Cursor::new(data);

        let mut reader = GGUFStreamReader::new(cursor).unwrap();

        // Read first tensor
        let result = reader.read_next_tensor().unwrap();
        assert!(result.is_some());
        let (name, tensor_data) = result.unwrap();
        assert_eq!(name, "tensor_a");
        assert_eq!(tensor_data.len(), 16);

        // Read second tensor
        let result = reader.read_next_tensor().unwrap();
        assert!(result.is_some());
        let (name, tensor_data) = result.unwrap();
        assert_eq!(name, "tensor_b");
        assert_eq!(tensor_data.len(), 12);

        // No more tensors
        let result = reader.read_next_tensor().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_read_all_tensors() {
        let data = create_stream_gguf_data();
        let cursor = Cursor::new(data);

        let mut reader = GGUFStreamReader::new(cursor).unwrap();
        let tensors = reader.read_all_tensors().unwrap();

        assert_eq!(tensors.len(), 2);
        assert!(tensors.contains_key("tensor_a"));
        assert!(tensors.contains_key("tensor_b"));
        assert_eq!(tensors["tensor_a"].len(), 16);
        assert_eq!(tensors["tensor_b"].len(), 12);
    }

    #[test]
    fn test_tensor_iterator() {
        let data = create_stream_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFStreamReader::new(cursor).unwrap();
        let mut iterator = reader.tensor_iterator();

        // First tensor
        let first = iterator.next().unwrap().unwrap();
        assert_eq!(first.0, "tensor_a");
        assert_eq!(first.1.len(), 16);

        // Second tensor
        let second = iterator.next().unwrap().unwrap();
        assert_eq!(second.0, "tensor_b");
        assert_eq!(second.1.len(), 12);

        // End of iteration
        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_stream_summary() {
        let data = create_stream_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFStreamReader::new(cursor).unwrap();
        let summary = reader.summary();

        assert_eq!(summary.tensor_count, 2);
        assert_eq!(summary.metadata_count, 1);
        assert_eq!(summary.total_tensor_size, 28); // 16 + 12 bytes
        assert!(summary.tensor_types.contains_key(&TensorType::F32));
    }

    #[test]
    fn test_stream_validation() {
        let data = create_stream_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFStreamReader::new(cursor).unwrap();
        assert!(reader.validate().is_ok());
    }

    #[test]
    fn test_limits_exceeded() {
        let data = create_stream_gguf_data();
        let cursor = Cursor::new(data);

        let config = StreamReaderConfig {
            max_tensor_count: 1, // Only allow 1 tensor, but we have 2
            ..Default::default()
        };

        let result = GGUFStreamReader::with_config(cursor, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_convenience_functions() {
        let data = create_stream_gguf_data();

        // Test stream_reader_from_read
        let cursor = Cursor::new(data.clone());
        let reader = stream_reader_from_read(cursor).unwrap();
        assert_eq!(reader.tensor_count(), 2);

        // Test buffered_stream_reader
        let cursor = Cursor::new(data);
        let reader = buffered_stream_reader(cursor).unwrap();
        assert_eq!(reader.tensor_count(), 2);
    }

    #[test]
    fn test_summary_display() {
        let data = create_stream_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFStreamReader::new(cursor).unwrap();
        let summary = reader.summary();
        let display_str = format!("{}", summary);

        assert!(display_str.contains("GGUF Stream Summary"));
        assert!(display_str.contains("Tensors: 2"));
        assert!(display_str.contains("F32"));
    }

    #[test]
    fn test_into_inner() {
        let data = create_stream_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFStreamReader::new(cursor).unwrap();
        let _inner = reader.into_inner(); // Should consume the reader successfully
    }
}
