//! File-based GGUF writer
//!
//! This module provides low-level functionality for writing GGUF files.
//! For most use cases, consider using the high-level `GGUFBuilder` instead.
//!
//! ## Example
//!
//! ```rust
//! # use gguf_rs::prelude::*;
//! # use gguf_rs::format::{GGUFHeader, Metadata};
//! # use gguf_rs::format::metadata::MetadataValue;
//! # use gguf_rs::tensor::{TensorInfo, TensorData, TensorShape, TensorType};
//! # fn main() -> Result<()> {
//! let mut buffer = Vec::new();
//! let mut writer = GGUFFileWriter::new(&mut buffer);
//!
//! // Create metadata
//! let mut metadata = Metadata::new();
//! metadata.insert("name".to_string(), MetadataValue::String("test".to_string()));
//!
//! // Create tensor data
//! let shape = TensorShape::new(vec![2, 2])?;
//! let tensor_info = TensorInfo::new("weights".to_string(), shape, TensorType::F32, 0);
//! let tensor_data = TensorData::new_owned(vec![0u8; 16]); // 4 F32 values
//! let tensors = vec![(tensor_info, tensor_data)];
//!
//! // Write complete file
//! let result = writer.write_complete_file(&metadata, &tensors)?;
//! println!("Wrote {} bytes", result.total_bytes_written);
//! # Ok(())
//! # }
//! ```

use crate::error::{GGUFError, Result};
use crate::format::{
    alignment::{align_to_default, AlignmentTracker},
    constants::GGUF_DEFAULT_ALIGNMENT,
    GGUFHeader, Metadata, TensorInfo,
};
use crate::tensor::{TensorData, TensorInfo as TensorInfoNew};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// A writer for GGUF files
#[derive(Debug)]
pub struct GGUFFileWriter<W> {
    /// The underlying writer
    writer: W,
    /// Current position in the file
    position: u64,
    /// Alignment tracker
    alignment_tracker: AlignmentTracker,
    /// Whether the header has been written
    header_written: bool,
    /// Whether we're in the tensor data section
    in_tensor_section: bool,
}

/// Configuration for GGUF file writing
#[derive(Debug, Clone)]
pub struct GGUFWriterConfig {
    /// Alignment for tensor data (default: 32 bytes)
    pub tensor_alignment: usize,
    /// Whether to validate data before writing
    pub validate_data: bool,
    /// Buffer size for writing
    pub buffer_size: usize,
    /// Whether to compute and store checksums
    pub compute_checksums: bool,
    /// Whether to compress metadata
    pub compress_metadata: bool,
}

impl Default for GGUFWriterConfig {
    fn default() -> Self {
        Self {
            tensor_alignment: GGUF_DEFAULT_ALIGNMENT,
            validate_data: true,
            buffer_size: 64 * 1024, // 64KB buffer
            compute_checksums: false,
            compress_metadata: false,
        }
    }
}

/// Information about what was written
#[derive(Debug, Clone)]
pub struct WriteResult {
    /// Number of bytes written
    pub bytes_written: usize,
    /// Final position in the file
    pub final_position: u64,
    /// Whether data was validated
    pub was_validated: bool,
    /// Checksum of written data (if computed)
    pub checksum: Option<u32>,
}

impl<W: Write> GGUFFileWriter<W> {
    /// Create a new GGUF file writer with default configuration
    pub fn new(writer: W) -> Self {
        Self::with_config(writer, GGUFWriterConfig::default())
    }

    /// Create a new GGUF file writer with custom configuration
    pub fn with_config(writer: W, config: GGUFWriterConfig) -> Self {
        Self {
            writer,
            position: 0,
            alignment_tracker: AlignmentTracker::new(config.tensor_alignment),
            header_written: false,
            in_tensor_section: false,
        }
    }

    /// Write the GGUF header
    pub fn write_header(&mut self, header: &GGUFHeader) -> Result<WriteResult> {
        if self.header_written {
            return Err(GGUFError::Format("Header already written".to_string()));
        }

        let _start_position = self.position;
        header.write_to(&mut self.writer)?;

        let bytes_written = GGUFHeader::size();
        self.position += bytes_written as u64;
        self.alignment_tracker.advance(bytes_written);
        self.header_written = true;

        Ok(WriteResult {
            bytes_written,
            final_position: self.position,
            was_validated: true, // Header validation is built-in
            checksum: None,
        })
    }

    /// Write metadata
    pub fn write_metadata(&mut self, metadata: &Metadata) -> Result<WriteResult> {
        if !self.header_written {
            return Err(GGUFError::Format("Header must be written before metadata".to_string()));
        }

        let _start_position = self.position;
        metadata.write_to(&mut self.writer)?;

        let bytes_written = metadata.serialized_size();
        self.position += bytes_written as u64;
        self.alignment_tracker.advance(bytes_written);

        Ok(WriteResult {
            bytes_written,
            final_position: self.position,
            was_validated: true,
            checksum: None,
        })
    }

    /// Write tensor information section
    pub fn write_tensor_infos(&mut self, tensor_infos: &[TensorInfoNew]) -> Result<WriteResult> {
        if !self.header_written {
            return Err(GGUFError::Format("Header must be written before tensor info".to_string()));
        }

        let _start_position = self.position;
        let mut total_bytes = 0;

        for tensor_info in tensor_infos {
            // Convert to the format's TensorInfo
            let format_tensor_info = TensorInfo::new(
                tensor_info.name().to_string(),
                tensor_info.shape().dims().to_vec(),
                tensor_info.tensor_type() as u32,
                tensor_info.data_offset(),
            );

            format_tensor_info.write_to(&mut self.writer)?;
            let info_size = format_tensor_info.serialized_size();
            total_bytes += info_size;
            self.position += info_size as u64;
            self.alignment_tracker.advance(info_size);
        }

        Ok(WriteResult {
            bytes_written: total_bytes,
            final_position: self.position,
            was_validated: true,
            checksum: None,
        })
    }

    /// Align to tensor data section
    pub fn align_for_tensor_data(&mut self) -> Result<WriteResult> {
        if self.in_tensor_section {
            return Err(GGUFError::Format("Already in tensor section".to_string()));
        }

        let alignment_info = self.alignment_tracker.align_default();

        if alignment_info.needs_padding() {
            let padding = alignment_info.padding_bytes();
            self.writer.write_all(&padding)?;
            self.position += padding.len() as u64;
        }

        self.in_tensor_section = true;

        Ok(WriteResult {
            bytes_written: alignment_info.padding,
            final_position: self.position,
            was_validated: false,
            checksum: None,
        })
    }

    /// Write tensor data
    pub fn write_tensor_data(
        &mut self,
        tensor_info: &TensorInfoNew,
        data: &TensorData,
    ) -> Result<WriteResult> {
        if !self.in_tensor_section {
            return Err(GGUFError::Format("Must align for tensor data first".to_string()));
        }

        // Validate data size matches expectation
        let expected_size = tensor_info.expected_data_size() as usize;
        if data.len() != expected_size {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' data size mismatch: expected {}, got {}",
                tensor_info.name(),
                expected_size,
                data.len()
            )));
        }

        let _start_position = self.position;
        let data_bytes = data.as_slice();

        // Compute checksum if requested
        let checksum = Some(data.checksum());

        // Write the data
        self.writer.write_all(data_bytes)?;

        let bytes_written = data_bytes.len();
        self.position += bytes_written as u64;
        self.alignment_tracker.advance(bytes_written);

        Ok(WriteResult {
            bytes_written,
            final_position: self.position,
            was_validated: true,
            checksum,
        })
    }

    /// Write multiple tensors in sequence
    pub fn write_multiple_tensors(
        &mut self,
        tensors: &[(TensorInfoNew, TensorData)],
    ) -> Result<Vec<WriteResult>> {
        let mut results = Vec::with_capacity(tensors.len());

        for (tensor_info, tensor_data) in tensors {
            let result = self.write_tensor_data(tensor_info, tensor_data)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Write a complete GGUF file
    pub fn write_complete_file(
        &mut self,
        metadata: &Metadata,
        tensors: &[(TensorInfoNew, TensorData)],
    ) -> Result<GGUFWriteResult> {
        // Create header
        let header = GGUFHeader::new(tensors.len() as u64, metadata.len() as u64);

        // Write header
        let header_result = self.write_header(&header)?;

        // Write metadata
        let metadata_result = self.write_metadata(metadata)?;

        // Calculate proper tensor data offsets
        // First, we need to predict where tensor data will start after tensor infos + alignment
        let mut predicted_position = self.position as usize;

        // Add size of tensor infos section
        for (tensor_info, _) in tensors {
            predicted_position += 8; // name length (u64)
            predicted_position += tensor_info.name().len(); // name bytes
            predicted_position += 4; // n_dimensions (u32)
            predicted_position += tensor_info.shape().dims().len() * 8; // dimensions (u64s)
            predicted_position += 4; // tensor_type (u32)
            predicted_position += 8; // data_offset (u64)
        }

        // Add alignment padding to reach tensor data section
        let _aligned_position = align_to_default(predicted_position);
        let mut current_data_offset = 0u64; // Tensor offsets are relative to tensor data start

        // Create tensor infos with correct offsets
        let mut tensor_infos_with_offsets = Vec::new();
        for (tensor_info, tensor_data) in tensors {
            let tensor_info_with_offset = TensorInfoNew::new(
                tensor_info.name().to_string(),
                tensor_info.shape().clone(),
                tensor_info.tensor_type(),
                current_data_offset,
            );
            tensor_infos_with_offsets.push(tensor_info_with_offset);
            current_data_offset += tensor_data.len() as u64;
        }

        // Write tensor infos with correct offsets
        let tensor_info_result = self.write_tensor_infos(&tensor_infos_with_offsets)?;

        // Align for tensor data
        let alignment_result = self.align_for_tensor_data()?;

        // Write tensor data
        let tensor_results = self.write_multiple_tensors(tensors)?;

        let total_bytes_written = header_result.bytes_written
            + metadata_result.bytes_written
            + tensor_info_result.bytes_written
            + alignment_result.bytes_written
            + tensor_results.iter().map(|r| r.bytes_written).sum::<usize>();

        Ok(GGUFWriteResult {
            header_result,
            metadata_result,
            tensor_info_result,
            alignment_result,
            tensor_results,
            total_bytes_written,
            final_position: self.position,
        })
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    /// Get current position
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Check if header has been written
    pub fn header_written(&self) -> bool {
        self.header_written
    }

    /// Check if in tensor section
    pub fn in_tensor_section(&self) -> bool {
        self.in_tensor_section
    }

    /// Get alignment tracker
    pub fn alignment_tracker(&self) -> &AlignmentTracker {
        &self.alignment_tracker
    }

    /// Finalize the file (flush and ensure all data is written)
    pub fn finalize(mut self) -> Result<W> {
        self.flush()?;
        Ok(self.writer)
    }
}

impl<W: Write + Seek> GGUFFileWriter<W> {
    /// Create a seekable writer
    pub fn with_seek(writer: W) -> Self {
        Self::new(writer)
    }

    /// Seek to a specific position
    pub fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let new_pos = self.writer.seek(pos)?;
        self.position = new_pos;
        self.alignment_tracker.position = new_pos as usize;
        Ok(new_pos)
    }

    /// Write tensor data at a specific position
    pub fn write_tensor_at_position(
        &mut self,
        tensor_info: &TensorInfoNew,
        data: &TensorData,
        position: u64,
    ) -> Result<WriteResult> {
        let original_pos = self.position;

        // Seek to target position
        self.seek(SeekFrom::Start(position))?;

        // Write tensor data
        let result = self.write_tensor_data(tensor_info, data)?;

        // Return to original position
        self.seek(SeekFrom::Start(original_pos))?;

        Ok(result)
    }

    /// Update tensor offsets after writing tensor info
    pub fn update_tensor_offsets(
        &mut self,
        tensor_infos: &mut [TensorInfoNew],
        tensor_info_start_position: u64,
    ) -> Result<()> {
        let current_pos = self.position;

        // Calculate where tensor data will start
        let _tensor_data_start = align_to_default(self.position as usize) as u64;
        let mut current_offset = 0u64;

        // Update each tensor's offset and rewrite the tensor info
        for (i, tensor_info) in tensor_infos.iter_mut().enumerate() {
            // Update the offset
            *tensor_info = TensorInfoNew::new(
                tensor_info.name().to_string(),
                tensor_info.shape().clone(),
                tensor_info.tensor_type(),
                current_offset,
            );

            // Seek to this tensor info's position in the file
            let info_position = tensor_info_start_position + (i as u64 * 64); // Approximate
            self.seek(SeekFrom::Start(info_position))?;

            // Rewrite the tensor info with updated offset
            let format_tensor_info = TensorInfo::new(
                tensor_info.name().to_string(),
                tensor_info.shape().dims().to_vec(),
                tensor_info.tensor_type() as u32,
                tensor_info.data_offset(),
            );
            format_tensor_info.write_to(&mut self.writer)?;

            // Calculate next offset
            current_offset += tensor_info.expected_data_size();
        }

        // Return to original position
        self.seek(SeekFrom::Start(current_pos))?;

        Ok(())
    }
}

/// Result of writing a complete GGUF file
#[derive(Debug, Clone)]
pub struct GGUFWriteResult {
    /// Result of writing header
    pub header_result: WriteResult,
    /// Result of writing metadata
    pub metadata_result: WriteResult,
    /// Result of writing tensor info
    pub tensor_info_result: WriteResult,
    /// Result of alignment padding
    pub alignment_result: WriteResult,
    /// Results of writing tensor data
    pub tensor_results: Vec<WriteResult>,
    /// Total bytes written
    pub total_bytes_written: usize,
    /// Final position in file
    pub final_position: u64,
}

impl GGUFWriteResult {
    /// Get total tensor data bytes written
    pub fn tensor_data_bytes(&self) -> usize {
        self.tensor_results.iter().map(|r| r.bytes_written).sum()
    }

    /// Get overhead bytes (non-tensor data)
    pub fn overhead_bytes(&self) -> usize {
        self.header_result.bytes_written
            + self.metadata_result.bytes_written
            + self.tensor_info_result.bytes_written
            + self.alignment_result.bytes_written
    }

    /// Get compression ratio (overhead / total)
    pub fn overhead_ratio(&self) -> f32 {
        if self.total_bytes_written == 0 {
            0.0
        } else {
            self.overhead_bytes() as f32 / self.total_bytes_written as f32
        }
    }
}

/// Convenience function to create a GGUF file at a path
pub fn create_gguf_file<P: AsRef<Path>>(
    path: P,
    metadata: &Metadata,
    tensors: &[(TensorInfoNew, TensorData)],
) -> Result<GGUFWriteResult> {
    let file = File::create(path)?;
    let buf_writer = BufWriter::new(file);
    let mut writer = GGUFFileWriter::new(buf_writer);

    writer.write_complete_file(metadata, tensors)
}

/// Convenience function to create a GGUF file with custom configuration
pub fn create_gguf_file_with_config<P: AsRef<Path>>(
    path: P,
    metadata: &Metadata,
    tensors: &[(TensorInfoNew, TensorData)],
    config: GGUFWriterConfig,
) -> Result<GGUFWriteResult> {
    let file = File::create(path)?;
    let buf_writer = BufWriter::new(file);
    let mut writer = GGUFFileWriter::with_config(buf_writer, config);

    writer.write_complete_file(metadata, tensors)
}

impl std::fmt::Display for WriteResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WriteResult {{ bytes: {}, pos: {}, validated: {}{}}}",
            self.bytes_written,
            self.final_position,
            self.was_validated,
            if let Some(checksum) = self.checksum {
                format!(", checksum: 0x{:08x}", checksum)
            } else {
                String::new()
            }
        )
    }
}

impl std::fmt::Display for GGUFWriteResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GGUF Write Result:")?;
        writeln!(f, "  Total bytes: {}", self.total_bytes_written)?;
        writeln!(f, "  Final position: {}", self.final_position)?;
        writeln!(f, "  Header: {} bytes", self.header_result.bytes_written)?;
        writeln!(f, "  Metadata: {} bytes", self.metadata_result.bytes_written)?;
        writeln!(f, "  Tensor info: {} bytes", self.tensor_info_result.bytes_written)?;
        writeln!(f, "  Alignment: {} bytes", self.alignment_result.bytes_written)?;
        writeln!(f, "  Tensor data: {} bytes", self.tensor_data_bytes())?;
        writeln!(f, "  Overhead ratio: {:.2}%", self.overhead_ratio() * 100.0)?;
        writeln!(f, "  Tensors written: {}", self.tensor_results.len())?;

        Ok(())
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::format::metadata::MetadataValue;
    use crate::tensor::{TensorShape, TensorType};

    fn create_test_metadata() -> Metadata {
        let mut metadata = Metadata::new();
        metadata.insert("name".to_string(), MetadataValue::String("test_model".to_string()));
        metadata.insert("version".to_string(), MetadataValue::U32(1));
        metadata
    }

    fn create_test_tensor() -> (TensorInfoNew, TensorData) {
        let shape = TensorShape::new(vec![2, 2]).unwrap();
        let tensor_info = TensorInfoNew::new("test_tensor".to_string(), shape, TensorType::F32, 0);
        let data = TensorData::new_owned(vec![0u8; 16]); // 4 F32 values
        (tensor_info, data)
    }

    #[test]
    fn test_writer_creation() {
        let buffer = Vec::new();
        let writer = GGUFFileWriter::new(buffer);

        assert_eq!(writer.position(), 0);
        assert!(!writer.header_written());
        assert!(!writer.in_tensor_section());
    }

    #[test]
    fn test_writer_with_config() {
        let buffer = Vec::new();
        let config =
            GGUFWriterConfig { tensor_alignment: 64, validate_data: false, ..Default::default() };
        let writer = GGUFFileWriter::with_config(buffer, config);

        assert_eq!(writer.alignment_tracker().default_alignment, 64);
    }

    #[test]
    fn test_write_header() {
        let buffer = Vec::new();
        let mut writer = GGUFFileWriter::new(buffer);

        let header = GGUFHeader::new(1, 1);
        let result = writer.write_header(&header).unwrap();

        assert_eq!(result.bytes_written, 24); // Header size
        assert!(writer.header_written());
        assert!(result.was_validated);
    }

    #[test]
    fn test_write_metadata() {
        let buffer = Vec::new();
        let mut writer = GGUFFileWriter::new(buffer);

        // Must write header first
        let header = GGUFHeader::new(1, 1);
        writer.write_header(&header).unwrap();

        let metadata = create_test_metadata();
        let result = writer.write_metadata(&metadata).unwrap();

        assert!(result.bytes_written > 0);
        assert!(result.was_validated);
    }

    #[test]
    fn test_write_complete_file() {
        let buffer = Vec::new();
        let mut writer = GGUFFileWriter::new(buffer);

        let metadata = create_test_metadata();
        let (tensor_info, tensor_data) = create_test_tensor();
        let tensors = vec![(tensor_info, tensor_data)];

        let result = writer.write_complete_file(&metadata, &tensors).unwrap();

        assert!(result.total_bytes_written > 0);
        assert_eq!(result.tensor_results.len(), 1);
        assert!(result.tensor_data_bytes() > 0);
        assert!(result.overhead_bytes() > 0);
    }

    #[test]
    fn test_write_order_enforcement() {
        let buffer = Vec::new();
        let mut writer = GGUFFileWriter::new(buffer);

        let metadata = create_test_metadata();

        // Try to write metadata before header - should fail
        let result = writer.write_metadata(&metadata);
        assert!(result.is_err());

        // Write header first
        let header = GGUFHeader::new(1, 1);
        writer.write_header(&header).unwrap();

        // Now metadata should work
        let result = writer.write_metadata(&metadata);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tensor_data_validation() {
        let buffer = Vec::new();
        let mut writer = GGUFFileWriter::new(buffer);

        let header = GGUFHeader::new(1, 1);
        writer.write_header(&header).unwrap();

        let metadata = create_test_metadata();
        writer.write_metadata(&metadata).unwrap();

        let shape = TensorShape::new(vec![2]).unwrap();
        let tensor_info = TensorInfoNew::new("test".to_string(), shape, TensorType::F32, 0);
        let tensor_infos = vec![tensor_info.clone()];
        writer.write_tensor_infos(&tensor_infos).unwrap();
        writer.align_for_tensor_data().unwrap();

        // Try to write wrong-sized data
        let wrong_data = TensorData::new_owned(vec![0u8; 4]); // Should be 8 bytes for 2 F32
        let result = writer.write_tensor_data(&tensor_info, &wrong_data);
        assert!(result.is_err());

        // Write correct-sized data
        let correct_data = TensorData::new_owned(vec![0u8; 8]);
        let result = writer.write_tensor_data(&tensor_info, &correct_data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_alignment() {
        let buffer = Vec::new();
        let mut writer = GGUFFileWriter::new(buffer);

        let header = GGUFHeader::new(1, 1);
        writer.write_header(&header).unwrap();

        let metadata = create_test_metadata();
        writer.write_metadata(&metadata).unwrap();

        let (tensor_info, _) = create_test_tensor();
        let tensor_infos = vec![tensor_info];
        writer.write_tensor_infos(&tensor_infos).unwrap();

        let pos_before = writer.position();
        let result = writer.align_for_tensor_data().unwrap();
        let pos_after = writer.position();

        // Position should be aligned to 32 bytes
        assert_eq!(pos_after % 32, 0);
        assert_eq!(result.bytes_written, (pos_after - pos_before) as usize);
    }

    #[test]
    fn test_multiple_tensors() {
        let buffer = Vec::new();
        let mut writer = GGUFFileWriter::new(buffer);

        let metadata = create_test_metadata();
        let (tensor1, data1) = create_test_tensor();
        let mut tensor2 = tensor1.clone();
        // Change name to make it different
        tensor2 = TensorInfoNew::new(
            "tensor2".to_string(),
            tensor2.shape().clone(),
            tensor2.tensor_type(),
            tensor2.data_offset(),
        );
        let data2 = data1.clone();

        let tensors = vec![(tensor1, data1), (tensor2, data2)];

        let result = writer.write_complete_file(&metadata, &tensors).unwrap();

        assert_eq!(result.tensor_results.len(), 2);
        assert!(result.total_bytes_written > 0);
    }

    #[test]
    fn test_convenience_functions() {
        use tempfile::NamedTempFile;

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let metadata = create_test_metadata();
        let (tensor_info, tensor_data) = create_test_tensor();
        let tensors = vec![(tensor_info, tensor_data)];

        let result = create_gguf_file(path, &metadata, &tensors).unwrap();
        assert!(result.total_bytes_written > 0);

        // File should exist and have content
        let file_size = std::fs::metadata(path).unwrap().len();
        assert!(file_size > 0);
    }

    #[test]
    fn test_display_implementations() {
        let write_result = WriteResult {
            bytes_written: 100,
            final_position: 200,
            was_validated: true,
            checksum: Some(0x12345678),
        };

        let display_str = format!("{}", write_result);
        assert!(display_str.contains("100"));
        assert!(display_str.contains("200"));
        assert!(display_str.contains("validated: true"));
        assert!(display_str.contains("0x12345678"));

        let gguf_result = GGUFWriteResult {
            header_result: WriteResult {
                bytes_written: 24,
                final_position: 24,
                was_validated: true,
                checksum: None,
            },
            metadata_result: write_result.clone(),
            tensor_info_result: write_result.clone(),
            alignment_result: WriteResult {
                bytes_written: 8,
                final_position: 332,
                was_validated: false,
                checksum: None,
            },
            tensor_results: vec![write_result],
            total_bytes_written: 432,
            final_position: 432,
        };

        let gguf_display = format!("{}", gguf_result);
        assert!(gguf_display.contains("432"));
        assert!(gguf_display.contains("GGUF Write Result"));
    }
}
