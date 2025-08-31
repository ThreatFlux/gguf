//! Stream-based GGUF writer for non-seekable streams

use crate::error::{GGUFError, Result};
use crate::format::{
    alignment::AlignmentTracker, constants::GGUF_DEFAULT_ALIGNMENT, GGUFHeader, Metadata,
    TensorInfo,
};
use crate::tensor::{TensorData, TensorInfo as TensorInfoNew};
use std::io::Write;

/// A writer for GGUF files to non-seekable streams
#[derive(Debug)]
pub struct GGUFStreamWriter<W> {
    /// The underlying writer
    writer: W,
    /// Current position in the stream
    position: u64,
    /// Alignment tracker
    alignment_tracker: AlignmentTracker,
    /// Configuration
    config: StreamWriterConfig,
    /// Write state
    state: WriterState,
}

/// Configuration for stream writing
#[derive(Debug, Clone)]
pub struct StreamWriterConfig {
    /// Tensor data alignment
    pub tensor_alignment: usize,
    /// Whether to validate data before writing
    pub validate_data: bool,
    /// Buffer size for internal operations
    pub buffer_size: usize,
}

impl Default for StreamWriterConfig {
    fn default() -> Self {
        Self {
            tensor_alignment: GGUF_DEFAULT_ALIGNMENT,
            validate_data: true,
            buffer_size: 64 * 1024,
        }
    }
}

/// Internal writer state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriterState {
    /// Ready to write header
    Ready,
    /// Header written, ready for metadata
    HeaderWritten,
    /// Metadata written, ready for tensor info
    MetadataWritten,
    /// Tensor info written, ready for alignment
    TensorInfoWritten,
    /// Aligned for tensor data, ready to write tensors
    TensorDataReady,
    /// Writing tensor data
    WritingTensors,
    /// Writing complete
    Finished,
}

/// Result of a stream write operation
#[derive(Debug, Clone)]
pub struct StreamWriteResult {
    /// Bytes written in this operation
    pub bytes_written: usize,
    /// Current position after write
    pub current_position: u64,
    /// Whether validation was performed
    pub validated: bool,
}

impl<W: Write> GGUFStreamWriter<W> {
    /// Create a new stream writer
    pub fn new(writer: W) -> Self {
        Self::with_config(writer, StreamWriterConfig::default())
    }

    /// Create a new stream writer with configuration
    pub fn with_config(writer: W, config: StreamWriterConfig) -> Self {
        Self {
            writer,
            position: 0,
            alignment_tracker: AlignmentTracker::new(config.tensor_alignment),
            config,
            state: WriterState::Ready,
        }
    }

    /// Write the header
    pub fn write_header(&mut self, header: &GGUFHeader) -> Result<StreamWriteResult> {
        if self.state != WriterState::Ready {
            return Err(GGUFError::Format("Header already written or invalid state".to_string()));
        }

        if self.config.validate_data {
            header.validate()?;
        }

        header.write_to(&mut self.writer)?;

        let bytes_written = GGUFHeader::size();
        self.position += bytes_written as u64;
        self.alignment_tracker.advance(bytes_written);
        self.state = WriterState::HeaderWritten;

        Ok(StreamWriteResult {
            bytes_written,
            current_position: self.position,
            validated: self.config.validate_data,
        })
    }

    /// Write metadata
    pub fn write_metadata(&mut self, metadata: &Metadata) -> Result<StreamWriteResult> {
        if self.state != WriterState::HeaderWritten {
            return Err(GGUFError::Format("Must write header before metadata".to_string()));
        }

        metadata.write_to(&mut self.writer)?;

        let bytes_written = metadata.serialized_size();
        self.position += bytes_written as u64;
        self.alignment_tracker.advance(bytes_written);
        self.state = WriterState::MetadataWritten;

        Ok(StreamWriteResult { bytes_written, current_position: self.position, validated: true })
    }

    /// Write tensor information
    pub fn write_tensor_infos(
        &mut self,
        tensor_infos: &[TensorInfoNew],
    ) -> Result<StreamWriteResult> {
        if self.state != WriterState::MetadataWritten {
            return Err(GGUFError::Format("Must write metadata before tensor info".to_string()));
        }

        let mut total_bytes = 0;
        let mut current_offset = 0u64;

        for tensor_info in tensor_infos {
            if self.config.validate_data {
                tensor_info.validate()?;
            }

            // Create tensor info with calculated offset
            let info_with_offset = TensorInfoNew::new(
                tensor_info.name().to_string(),
                tensor_info.shape().clone(),
                tensor_info.tensor_type(),
                current_offset,
            );

            // Convert to format TensorInfo and write
            let format_tensor_info = TensorInfo::new(
                info_with_offset.name().to_string(),
                info_with_offset.shape().dims().to_vec(),
                info_with_offset.tensor_type() as u32,
                info_with_offset.data_offset(),
            );

            format_tensor_info.write_to(&mut self.writer)?;
            let info_size = format_tensor_info.serialized_size();
            total_bytes += info_size;

            // Calculate next tensor's offset
            current_offset += tensor_info.expected_data_size();
        }

        self.position += total_bytes as u64;
        self.alignment_tracker.advance(total_bytes);
        self.state = WriterState::TensorInfoWritten;

        Ok(StreamWriteResult {
            bytes_written: total_bytes,
            current_position: self.position,
            validated: self.config.validate_data,
        })
    }

    /// Align for tensor data section
    pub fn align_for_tensor_data(&mut self) -> Result<StreamWriteResult> {
        if self.state != WriterState::TensorInfoWritten {
            return Err(GGUFError::Format("Must write tensor info before alignment".to_string()));
        }

        let alignment_info = self.alignment_tracker.align_default();

        let bytes_written = if alignment_info.needs_padding() {
            let padding = alignment_info.padding_bytes();
            self.writer.write_all(&padding)?;
            self.position += padding.len() as u64;
            padding.len()
        } else {
            0
        };

        self.state = WriterState::TensorDataReady;

        Ok(StreamWriteResult { bytes_written, current_position: self.position, validated: false })
    }

    /// Write tensor data (must be called in order)
    pub fn write_tensor_data(
        &mut self,
        tensor_info: &TensorInfoNew,
        data: &TensorData,
    ) -> Result<StreamWriteResult> {
        if !matches!(self.state, WriterState::TensorDataReady | WriterState::WritingTensors) {
            return Err(GGUFError::Format("Must align for tensor data first".to_string()));
        }

        if self.config.validate_data {
            let expected_size = tensor_info.expected_data_size() as usize;
            if data.len() != expected_size {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' size mismatch: expected {}, got {}",
                    tensor_info.name(),
                    expected_size,
                    data.len()
                )));
            }
        }

        let data_bytes = data.as_slice();
        self.writer.write_all(data_bytes)?;

        let bytes_written = data_bytes.len();
        self.position += bytes_written as u64;
        self.alignment_tracker.advance(bytes_written);
        self.state = WriterState::WritingTensors;

        Ok(StreamWriteResult {
            bytes_written,
            current_position: self.position,
            validated: self.config.validate_data,
        })
    }

    /// Write tensor data in chunks (for large tensors)
    pub fn write_tensor_data_chunked<R: std::io::Read>(
        &mut self,
        tensor_info: &TensorInfoNew,
        mut reader: R,
    ) -> Result<StreamWriteResult> {
        if !matches!(self.state, WriterState::TensorDataReady | WriterState::WritingTensors) {
            return Err(GGUFError::Format("Must align for tensor data first".to_string()));
        }

        let expected_size = tensor_info.expected_data_size() as usize;
        let mut buffer = vec![0u8; self.config.buffer_size.min(expected_size)];
        let mut total_written = 0;

        while total_written < expected_size {
            let to_read = (expected_size - total_written).min(buffer.len());
            buffer.resize(to_read, 0);

            reader.read_exact(&mut buffer)?;
            self.writer.write_all(&buffer)?;

            total_written += to_read;
        }

        self.position += total_written as u64;
        self.alignment_tracker.advance(total_written);
        self.state = WriterState::WritingTensors;

        Ok(StreamWriteResult {
            bytes_written: total_written,
            current_position: self.position,
            validated: self.config.validate_data,
        })
    }

    /// Write a complete GGUF file to stream
    pub fn write_complete_stream(
        &mut self,
        metadata: &Metadata,
        tensors: &[(TensorInfoNew, TensorData)],
    ) -> Result<CompleteStreamWriteResult> {
        // Write header
        let header = GGUFHeader::new(tensors.len() as u64, metadata.len() as u64);
        let header_result = self.write_header(&header)?;

        // Write metadata
        let metadata_result = self.write_metadata(metadata)?;

        // Write tensor infos
        let tensor_infos: Vec<TensorInfoNew> =
            tensors.iter().map(|(info, _)| info.clone()).collect();
        let tensor_info_result = self.write_tensor_infos(&tensor_infos)?;

        // Align for tensor data
        let alignment_result = self.align_for_tensor_data()?;

        // Write tensor data
        let mut tensor_results = Vec::new();
        for (tensor_info, tensor_data) in tensors {
            let result = self.write_tensor_data(tensor_info, tensor_data)?;
            tensor_results.push(result);
        }

        self.state = WriterState::Finished;

        let total_bytes = header_result.bytes_written
            + metadata_result.bytes_written
            + tensor_info_result.bytes_written
            + alignment_result.bytes_written
            + tensor_results.iter().map(|r| r.bytes_written).sum::<usize>();

        Ok(CompleteStreamWriteResult {
            header_result,
            metadata_result,
            tensor_info_result,
            alignment_result,
            tensor_results,
            total_bytes_written: total_bytes,
            final_position: self.position,
        })
    }

    /// Finalize the stream (flush and mark as finished)
    pub fn finalize(&mut self) -> Result<()> {
        self.writer.flush()?;
        if matches!(self.state, WriterState::WritingTensors) {
            self.state = WriterState::Finished;
        }
        Ok(())
    }

    /// Get current position
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Get current state
    pub fn state(&self) -> WriterState {
        self.state
    }

    /// Check if writing is complete
    pub fn is_finished(&self) -> bool {
        self.state == WriterState::Finished
    }

    /// Get the underlying writer
    pub fn into_inner(self) -> W {
        self.writer
    }
}

/// Result of writing a complete GGUF stream
#[derive(Debug, Clone)]
pub struct CompleteStreamWriteResult {
    /// Header write result
    pub header_result: StreamWriteResult,
    /// Metadata write result
    pub metadata_result: StreamWriteResult,
    /// Tensor info write result
    pub tensor_info_result: StreamWriteResult,
    /// Alignment write result
    pub alignment_result: StreamWriteResult,
    /// Tensor data write results
    pub tensor_results: Vec<StreamWriteResult>,
    /// Total bytes written
    pub total_bytes_written: usize,
    /// Final position
    pub final_position: u64,
}

impl CompleteStreamWriteResult {
    /// Get total tensor data bytes
    pub fn tensor_data_bytes(&self) -> usize {
        self.tensor_results.iter().map(|r| r.bytes_written).sum()
    }

    /// Get overhead bytes
    pub fn overhead_bytes(&self) -> usize {
        self.header_result.bytes_written
            + self.metadata_result.bytes_written
            + self.tensor_info_result.bytes_written
            + self.alignment_result.bytes_written
    }
}

/// Utility for streaming GGUF creation
pub struct StreamingGGUFBuilder<W> {
    writer: GGUFStreamWriter<W>,
    tensors_to_write: Vec<(TensorInfoNew, TensorData)>,
    metadata: Metadata,
}

impl<W: Write> StreamingGGUFBuilder<W> {
    /// Create a new streaming builder
    pub fn new(writer: W) -> Self {
        Self {
            writer: GGUFStreamWriter::new(writer),
            tensors_to_write: Vec::new(),
            metadata: Metadata::new(),
        }
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: crate::format::metadata::MetadataValue) {
        self.metadata.insert(key, value);
    }

    /// Add a tensor
    pub fn add_tensor(&mut self, tensor_info: TensorInfoNew, data: TensorData) {
        self.tensors_to_write.push((tensor_info, data));
    }

    /// Build and write the complete GGUF file
    pub fn build(mut self) -> Result<CompleteStreamWriteResult> {
        self.writer.write_complete_stream(&self.metadata, &self.tensors_to_write)
    }

    /// Get the number of tensors added
    pub fn tensor_count(&self) -> usize {
        self.tensors_to_write.len()
    }

    /// Get the metadata size
    pub fn metadata_size(&self) -> usize {
        self.metadata.len()
    }
}

impl std::fmt::Display for StreamWriteResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StreamWriteResult {{ bytes: {}, pos: {}, validated: {} }}",
            self.bytes_written, self.current_position, self.validated
        )
    }
}

impl std::fmt::Display for CompleteStreamWriteResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Complete Stream Write Result:")?;
        writeln!(f, "  Total bytes: {}", self.total_bytes_written)?;
        writeln!(f, "  Final position: {}", self.final_position)?;
        writeln!(f, "  Overhead: {} bytes", self.overhead_bytes())?;
        writeln!(f, "  Tensor data: {} bytes", self.tensor_data_bytes())?;
        writeln!(f, "  Tensors: {}", self.tensor_results.len())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::metadata::MetadataValue;
    use crate::tensor::{TensorShape, TensorType};

    fn create_test_setup() -> (Metadata, Vec<(TensorInfoNew, TensorData)>) {
        let mut metadata = Metadata::new();
        metadata.insert("name".to_string(), MetadataValue::String("test".to_string()));

        let shape = TensorShape::new(vec![2, 2]).unwrap();
        let tensor_info = TensorInfoNew::new("tensor".to_string(), shape, TensorType::F32, 0);
        let data = TensorData::new_owned(vec![0u8; 16]);

        (metadata, vec![(tensor_info, data)])
    }

    #[test]
    fn test_stream_writer_creation() {
        let buffer = Vec::new();
        let writer = GGUFStreamWriter::new(buffer);

        assert_eq!(writer.position(), 0);
        assert_eq!(writer.state(), WriterState::Ready);
        assert!(!writer.is_finished());
    }

    #[test]
    fn test_stream_writer_states() {
        let buffer = Vec::new();
        let mut writer = GGUFStreamWriter::new(buffer);

        // Initial state
        assert_eq!(writer.state(), WriterState::Ready);

        // Write header
        let header = GGUFHeader::new(1, 1);
        writer.write_header(&header).unwrap();
        assert_eq!(writer.state(), WriterState::HeaderWritten);

        // Write metadata
        let (metadata, _) = create_test_setup();
        writer.write_metadata(&metadata).unwrap();
        assert_eq!(writer.state(), WriterState::MetadataWritten);
    }

    #[test]
    fn test_write_complete_stream() {
        let buffer = Vec::new();
        let mut writer = GGUFStreamWriter::new(buffer);

        let (metadata, tensors) = create_test_setup();
        let result = writer.write_complete_stream(&metadata, &tensors).unwrap();

        assert!(result.total_bytes_written > 0);
        assert_eq!(result.tensor_results.len(), 1);
        assert!(writer.is_finished());
    }

    #[test]
    fn test_invalid_state_transitions() {
        let buffer = Vec::new();
        let mut writer = GGUFStreamWriter::new(buffer);

        let (metadata, _) = create_test_setup();

        // Try to write metadata before header
        let result = writer.write_metadata(&metadata);
        assert!(result.is_err());

        // Write header first
        let header = GGUFHeader::new(1, 1);
        writer.write_header(&header).unwrap();

        // Now metadata should work
        assert!(writer.write_metadata(&metadata).is_ok());
    }

    #[test]
    fn test_streaming_builder() {
        let buffer = Vec::new();
        let mut builder = StreamingGGUFBuilder::new(buffer);

        builder.add_metadata("test".to_string(), MetadataValue::U32(42));

        let shape = TensorShape::new(vec![4]).unwrap();
        let tensor_info = TensorInfoNew::new("tensor".to_string(), shape, TensorType::F32, 0);
        let data = TensorData::new_owned(vec![0u8; 16]);
        builder.add_tensor(tensor_info, data);

        assert_eq!(builder.tensor_count(), 1);
        assert_eq!(builder.metadata_size(), 1);

        let result = builder.build().unwrap();
        assert!(result.total_bytes_written > 0);
    }

    #[test]
    fn test_tensor_validation() {
        let buffer = Vec::new();
        let config = StreamWriterConfig { validate_data: true, ..Default::default() };
        let mut writer = GGUFStreamWriter::with_config(buffer, config);

        let (metadata, _) = create_test_setup();

        // Set up writer state
        let header = GGUFHeader::new(1, 1);
        writer.write_header(&header).unwrap();
        writer.write_metadata(&metadata).unwrap();

        let shape = TensorShape::new(vec![2]).unwrap();
        let tensor_info = TensorInfoNew::new("test".to_string(), shape, TensorType::F32, 0);
        writer.write_tensor_infos(&vec![tensor_info.clone()]).unwrap();
        writer.align_for_tensor_data().unwrap();

        // Try wrong-sized data
        let wrong_data = TensorData::new_owned(vec![0u8; 4]); // Should be 8 bytes
        let result = writer.write_tensor_data(&tensor_info, &wrong_data);
        assert!(result.is_err());

        // Correct size should work
        let correct_data = TensorData::new_owned(vec![0u8; 8]);
        let result = writer.write_tensor_data(&tensor_info, &correct_data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_chunked_writing() {
        use std::io::Cursor;

        let buffer = Vec::new();
        let mut writer = GGUFStreamWriter::new(buffer);

        let (metadata, _) = create_test_setup();

        // Set up for tensor writing
        let header = GGUFHeader::new(1, 1);
        writer.write_header(&header).unwrap();
        writer.write_metadata(&metadata).unwrap();

        let shape = TensorShape::new(vec![4]).unwrap();
        let tensor_info = TensorInfoNew::new("test".to_string(), shape, TensorType::F32, 0);
        writer.write_tensor_infos(&vec![tensor_info.clone()]).unwrap();
        writer.align_for_tensor_data().unwrap();

        // Write tensor data in chunks
        let data = vec![0u8; 16];
        let cursor = Cursor::new(data);
        let result = writer.write_tensor_data_chunked(&tensor_info, cursor).unwrap();

        assert_eq!(result.bytes_written, 16);
        assert!(result.validated);
    }

    #[test]
    fn test_display_implementations() {
        let stream_result =
            StreamWriteResult { bytes_written: 100, current_position: 200, validated: true };

        let display_str = format!("{}", stream_result);
        assert!(display_str.contains("100"));
        assert!(display_str.contains("200"));
        assert!(display_str.contains("validated: true"));

        let complete_result = CompleteStreamWriteResult {
            header_result: stream_result.clone(),
            metadata_result: stream_result.clone(),
            tensor_info_result: stream_result.clone(),
            alignment_result: StreamWriteResult {
                bytes_written: 8,
                current_position: 308,
                validated: false,
            },
            tensor_results: vec![stream_result],
            total_bytes_written: 408,
            final_position: 408,
        };

        let complete_display = format!("{}", complete_result);
        assert!(complete_display.contains("408"));
        assert!(complete_display.contains("Complete Stream Write Result"));
    }
}
