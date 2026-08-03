//! File-based GGUF writer
//!
//! This module provides low-level functionality for writing GGUF files.
//! For most use cases, consider using the high-level `GGUFBuilder` instead.
//!
//! ## Example
//!
//! ```rust
//! # use gguf_rs_lib::prelude::*;
//! # use gguf_rs_lib::format::{GGUFHeader, Metadata};
//! # use gguf_rs_lib::format::metadata::MetadataValue;
//! # use gguf_rs_lib::tensor::{TensorInfo, TensorData, TensorShape, TensorType};
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
    alignment::{AlignmentInfo, AlignmentTracker},
    constants::GGUF_DEFAULT_ALIGNMENT,
    metadata::MetadataValue,
    GGUFHeader, Metadata, TensorInfo,
};
use crate::tensor::{TensorData, TensorInfo as TensorInfoNew, TensorShape};
use std::collections::HashSet;
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
    /// Whether the declared metadata section has been written
    metadata_written: bool,
    /// Whether the declared tensor descriptor section has been written
    tensor_infos_written: bool,
    /// Tensor count declared by the header
    expected_tensor_count: u64,
    /// Metadata count declared by the header
    expected_metadata_count: u64,
    /// Whether we're in the tensor data section
    in_tensor_section: bool,
    /// Writer behavior and declared tensor alignment
    config: GGUFWriterConfig,
    /// Tensor descriptors persisted in their declared payload order
    declared_tensors: Vec<DeclaredTensor>,
    /// Index of the next tensor payload that must be written
    next_tensor: usize,
    /// Absolute start of the tensor data section
    tensor_data_start: Option<u64>,
}

#[derive(Debug)]
struct DeclaredTensor {
    descriptor: TensorInfo,
    expected_size: usize,
}

#[derive(Debug)]
struct PreparedCompleteFile {
    header: GGUFHeader,
    metadata: Metadata,
    tensor_infos: Vec<TensorInfoNew>,
}

impl DeclaredTensor {
    fn matches(&self, tensor_info: &TensorInfoNew) -> bool {
        self.descriptor.name == tensor_info.name()
            && self.descriptor.dimensions == tensor_info.shape().dims()
            && self.descriptor.tensor_type == tensor_info.tensor_type() as u32
            && self.descriptor.offset == tensor_info.data_offset()
    }
}

/// Configuration for GGUF file writing
#[derive(Debug, Clone)]
pub struct GGUFWriterConfig {
    /// Alignment for tensor data (default: 32 bytes)
    pub tensor_alignment: usize,
    /// Request optional content validation beyond mandatory format invariants.
    ///
    /// Descriptor validity and exact payload lengths are always enforced.
    pub validate_data: bool,
    /// Reserved buffer-size hint for future chunked file writes.
    ///
    /// The current writer delegates buffering to its underlying `Write` implementation.
    pub buffer_size: usize,
    /// Whether to compute a checksum returned in each [`WriteResult`].
    pub compute_checksums: bool,
    /// Request metadata compression.
    ///
    /// GGUF v3 has no standard compressed metadata representation; setting this
    /// to `true` returns [`GGUFError::FeatureUnavailable`].
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
            metadata_written: false,
            tensor_infos_written: false,
            expected_tensor_count: 0,
            expected_metadata_count: 0,
            in_tensor_section: false,
            config,
            declared_tensors: Vec::new(),
            next_tensor: 0,
            tensor_data_start: None,
        }
    }

    /// Write the GGUF header
    pub fn write_header(&mut self, header: &GGUFHeader) -> Result<WriteResult> {
        if self.header_written {
            return Err(GGUFError::Format("Header already written".to_string()));
        }

        self.validate_config()?;
        header.validate_comprehensive()?;
        header.write_to(&mut self.writer)?;

        let bytes_written = GGUFHeader::size();
        self.advance(bytes_written)?;
        self.header_written = true;
        self.expected_tensor_count = header.tensor_count;
        self.expected_metadata_count = header.metadata_kv_count;

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
        if self.metadata_written {
            return Err(GGUFError::Format("Metadata already written".to_string()));
        }
        if u64::try_from(metadata.len()).ok() != Some(self.expected_metadata_count) {
            return Err(GGUFError::InvalidMetadata(format!(
                "Header declares {} metadata entries, got {}",
                self.expected_metadata_count,
                metadata.len()
            )));
        }

        self.validate_metadata_alignment(metadata)?;
        metadata.write_to(&mut self.writer)?;

        let bytes_written = metadata.checked_serialized_size().ok_or_else(|| {
            GGUFError::InvalidMetadata("Serialized metadata size overflows usize".to_string())
        })?;
        self.advance(bytes_written)?;
        self.metadata_written = true;

        Ok(WriteResult {
            bytes_written,
            final_position: self.position,
            was_validated: true,
            checksum: None,
        })
    }

    /// Write tensor information section
    pub fn write_tensor_infos(&mut self, tensor_infos: &[TensorInfoNew]) -> Result<WriteResult> {
        if !self.metadata_written {
            return Err(GGUFError::Format(
                "Metadata must be written before tensor info".to_string(),
            ));
        }
        if self.tensor_infos_written {
            return Err(GGUFError::Format("Tensor information already written".to_string()));
        }
        if u64::try_from(tensor_infos.len()).ok() != Some(self.expected_tensor_count) {
            return Err(GGUFError::InvalidTensorData(format!(
                "Header declares {} tensors, got {} descriptors",
                self.expected_tensor_count,
                tensor_infos.len()
            )));
        }

        let declared_tensors =
            prepare_declared_tensors(tensor_infos, self.config.tensor_alignment)?;
        let mut total_bytes: usize = 0;
        for declared in &declared_tensors {
            declared.descriptor.write_to(&mut self.writer)?;
            let info_size = declared.descriptor.checked_serialized_size().ok_or_else(|| {
                GGUFError::InvalidTensorData("Tensor descriptor size overflows usize".to_string())
            })?;
            total_bytes = total_bytes.checked_add(info_size).ok_or_else(|| {
                GGUFError::InvalidTensorData(
                    "Tensor descriptor section overflows usize".to_string(),
                )
            })?;
            self.advance(info_size)?;
        }
        self.declared_tensors = declared_tensors;
        self.next_tensor = 0;
        self.tensor_infos_written = true;

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
        if !self.tensor_infos_written {
            return Err(GGUFError::Format(
                "Tensor information must be written before alignment".to_string(),
            ));
        }

        self.validate_config()?;
        let alignment_info =
            AlignmentInfo::new(self.alignment_tracker.position, self.config.tensor_alignment);

        if alignment_info.needs_padding() {
            self.write_padding(alignment_info.padding)?;
        }

        self.in_tensor_section = true;
        self.tensor_data_start = Some(self.position);

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

        let (expected_offset, expected_size) = {
            let expected = self.declared_tensors.get(self.next_tensor).ok_or_else(|| {
                GGUFError::InvalidTensorData(
                    "All declared tensor payloads are already written".to_string(),
                )
            })?;
            if !expected.matches(tensor_info) {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Expected payload descriptor '{}' at index {}, got '{}'",
                    expected.descriptor.name,
                    self.next_tensor,
                    tensor_info.name()
                )));
            }
            (expected.descriptor.offset, expected.expected_size)
        };
        if data.len() != expected_size {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' data size mismatch: expected {}, got {}",
                tensor_info.name(),
                expected_size,
                data.len()
            )));
        }

        let data_bytes = data.try_as_slice()?;
        let tensor_data_start = self.tensor_data_start.ok_or_else(|| {
            GGUFError::Format("Tensor data section start is unavailable".to_string())
        })?;
        let target = tensor_data_start.checked_add(expected_offset).ok_or_else(|| {
            GGUFError::InvalidTensorData("Tensor absolute offset overflows u64".to_string())
        })?;
        if self.position > target {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' offset precedes the current write position",
                tensor_info.name()
            )));
        }
        let padding = usize::try_from(target - self.position).map_err(|_| {
            GGUFError::InvalidTensorData("Tensor padding does not fit usize".to_string())
        })?;
        self.write_padding(padding)?;

        // Compute checksum if requested
        let checksum = self.config.compute_checksums.then(|| data.checksum());

        // Write the data
        self.writer.write_all(data_bytes)?;

        let bytes_written = data_bytes.len();
        self.advance(bytes_written)?;
        self.next_tensor = self.next_tensor.checked_add(1).ok_or_else(|| {
            GGUFError::InvalidTensorData("Tensor payload cursor overflows usize".to_string())
        })?;

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
        let mut results = Vec::new();
        results.try_reserve_exact(tensors.len()).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor write results".to_string())
        })?;

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
        let prepared = self.prepare_complete_file(metadata, tensors)?;
        self.write_prepared_complete_file(prepared, tensors)
    }

    fn prepare_complete_file(
        &self,
        metadata: &Metadata,
        tensors: &[(TensorInfoNew, TensorData)],
    ) -> Result<PreparedCompleteFile> {
        self.validate_config()?;
        let metadata = self.metadata_with_declared_alignment(metadata)?;
        let tensor_count = u64::try_from(tensors.len())
            .map_err(|_| GGUFError::Format("Tensor count does not fit u64".to_string()))?;
        let metadata_count = u64::try_from(metadata.len())
            .map_err(|_| GGUFError::Format("Metadata count does not fit u64".to_string()))?;
        let header = GGUFHeader::new(tensor_count, metadata_count);

        let mut current_data_offset = 0u64; // Tensor offsets are relative to tensor data start

        // Validate the complete caller-owned model before writing any bytes.
        // I/O failures can still leave partial output, but invalid descriptors,
        // duplicate names, and bad payloads fail without corrupting the target.
        let mut tensor_infos_with_offsets = Vec::new();
        tensor_infos_with_offsets.try_reserve_exact(tensors.len()).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor descriptor list".to_string())
        })?;
        let mut tensor_names = HashSet::new();
        tensor_names.try_reserve(tensors.len()).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor-name set".to_string())
        })?;
        for (tensor_info, tensor_data) in tensors {
            tensor_info.validate()?;
            tensor_data.validate()?;
            if !tensor_names.insert(tensor_info.name()) {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Duplicate tensor name: {}",
                    tensor_info.name()
                )));
            }
            let expected_size = tensor_info.checked_expected_data_size()?;
            if u64::try_from(tensor_data.len()).ok() != Some(expected_size) {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' data size mismatch: expected {}, got {}",
                    tensor_info.name(),
                    expected_size,
                    tensor_data.len()
                )));
            }
            current_data_offset =
                checked_align_u64(current_data_offset, self.config.tensor_alignment as u64)?;
            let tensor_info_with_offset =
                clone_tensor_info_with_offset(tensor_info, current_data_offset)?;
            tensor_infos_with_offsets.push(tensor_info_with_offset);
            current_data_offset =
                current_data_offset.checked_add(expected_size).ok_or_else(|| {
                    GGUFError::InvalidTensorData("Tensor data offsets overflow u64".to_string())
                })?;
        }

        Ok(PreparedCompleteFile { header, metadata, tensor_infos: tensor_infos_with_offsets })
    }

    fn write_prepared_complete_file(
        &mut self,
        prepared: PreparedCompleteFile,
        tensors: &[(TensorInfoNew, TensorData)],
    ) -> Result<GGUFWriteResult> {
        let PreparedCompleteFile { header, metadata, tensor_infos } = prepared;

        // Write only after all format-level input validation has succeeded.
        let header_result = self.write_header(&header)?;
        let metadata_result = self.write_metadata(&metadata)?;

        // Write tensor infos with correct offsets
        let tensor_info_result = self.write_tensor_infos(&tensor_infos)?;

        // Align for tensor data
        let alignment_result = self.align_for_tensor_data()?;

        // Write tensor data
        let mut tensor_results = Vec::new();
        tensor_results.try_reserve_exact(tensors.len()).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor write results".to_string())
        })?;
        for ((_, tensor_data), tensor_info) in tensors.iter().zip(tensor_infos.iter()) {
            tensor_results.push(self.write_tensor_data(tensor_info, tensor_data)?);
        }

        self.ensure_all_tensor_payloads_written()?;
        self.flush()?;

        let total_bytes_written = usize::try_from(self.position).map_err(|_| {
            GGUFError::Format("Final writer position does not fit usize".to_string())
        })?;

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

    fn validate_config(&self) -> Result<()> {
        if self.config.compress_metadata {
            return Err(GGUFError::FeatureUnavailable("compressed GGUF metadata".to_string()));
        }
        if self.config.tensor_alignment == 0 || !self.config.tensor_alignment.is_multiple_of(8) {
            return Err(GGUFError::InvalidMetadata(format!(
                "Tensor alignment must be a non-zero multiple of 8, got {}",
                self.config.tensor_alignment
            )));
        }
        u32::try_from(self.config.tensor_alignment).map_err(|_| {
            GGUFError::InvalidMetadata("Tensor alignment exceeds u32::MAX".to_string())
        })?;
        Ok(())
    }

    fn metadata_with_declared_alignment(&self, metadata: &Metadata) -> Result<Metadata> {
        let mut metadata = metadata.clone();
        match metadata.get("general.alignment") {
            Some(MetadataValue::U32(value))
                if usize::try_from(*value).ok() == Some(self.config.tensor_alignment) => {}
            Some(MetadataValue::U32(value)) => {
                return Err(GGUFError::InvalidMetadata(format!(
                    "general.alignment {} does not match writer alignment {}",
                    value, self.config.tensor_alignment
                )));
            }
            Some(value) => {
                return Err(GGUFError::InvalidMetadata(format!(
                    "general.alignment must be u32, got {}",
                    value.value_type()
                )));
            }
            None if self.config.tensor_alignment != GGUF_DEFAULT_ALIGNMENT => {
                metadata.insert(
                    "general.alignment".to_string(),
                    MetadataValue::U32(self.config.tensor_alignment as u32),
                );
            }
            None => {}
        }
        metadata.validate()?;
        Ok(metadata)
    }

    fn validate_metadata_alignment(&self, metadata: &Metadata) -> Result<()> {
        let declared = metadata.tensor_alignment()?;
        if declared != self.config.tensor_alignment {
            return Err(GGUFError::InvalidMetadata(format!(
                "Metadata alignment {} does not match writer alignment {}",
                declared, self.config.tensor_alignment
            )));
        }
        Ok(())
    }

    fn write_padding(&mut self, mut size: usize) -> Result<()> {
        const ZEROES: [u8; 8192] = [0; 8192];
        let original_size = size;
        while size > 0 {
            let chunk = size.min(ZEROES.len());
            self.writer.write_all(&ZEROES[..chunk])?;
            size -= chunk;
        }
        self.advance(original_size)
    }

    fn advance(&mut self, bytes: usize) -> Result<()> {
        let bytes_u64 = u64::try_from(bytes)
            .map_err(|_| GGUFError::Format("Write size does not fit u64".to_string()))?;
        self.position = self
            .position
            .checked_add(bytes_u64)
            .ok_or_else(|| GGUFError::Format("Writer position overflows u64".to_string()))?;
        self.alignment_tracker.checked_advance(bytes).ok_or_else(|| {
            GGUFError::Format("Writer alignment position overflows usize".to_string())
        })
    }

    fn ensure_all_tensor_payloads_written(&self) -> Result<()> {
        if !self.in_tensor_section || self.tensor_data_start.is_none() {
            return Err(GGUFError::Format("Tensor data section has not been started".to_string()));
        }
        if self.next_tensor != self.declared_tensors.len() {
            return Err(GGUFError::InvalidTensorData(format!(
                "Cannot finalize: wrote {} of {} declared tensor payloads",
                self.next_tensor,
                self.declared_tensors.len()
            )));
        }
        Ok(())
    }

    /// Finalize the file (flush and ensure all declared tensor data is written)
    pub fn finalize(mut self) -> Result<W> {
        self.ensure_all_tensor_payloads_written()?;
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
        self.alignment_tracker.position = usize::try_from(new_pos).map_err(|_| {
            GGUFError::Format("Seek position does not fit this platform".to_string())
        })?;
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
    #[deprecated(
        note = "descriptor rewriting is unsupported; compute aligned offsets before writing infos"
    )]
    pub fn update_tensor_offsets(
        &mut self,
        _tensor_infos: &mut [TensorInfoNew],
        _tensor_info_start_position: u64,
    ) -> Result<()> {
        Err(GGUFError::FeatureUnavailable("in-place tensor offset rewriting".to_string()))
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
        self.tensor_results
            .iter()
            .fold(0usize, |total, result| total.saturating_add(result.bytes_written))
    }

    /// Get overhead bytes (non-tensor data)
    pub fn overhead_bytes(&self) -> usize {
        self.total_bytes_written.saturating_sub(self.tensor_data_bytes())
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

fn checked_align_u64(position: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 {
        return Err(GGUFError::InvalidMetadata("Tensor alignment cannot be zero".to_string()));
    }
    let remainder = position % alignment;
    let padding = if remainder == 0 { 0 } else { alignment - remainder };
    position
        .checked_add(padding)
        .ok_or_else(|| GGUFError::Format("Tensor offset overflows u64".to_string()))
}

fn prepare_declared_tensors(
    tensor_infos: &[TensorInfoNew],
    alignment: usize,
) -> Result<Vec<DeclaredTensor>> {
    let alignment = u64::try_from(alignment)
        .map_err(|_| GGUFError::InvalidMetadata("Tensor alignment does not fit u64".to_string()))?;
    let mut declared = Vec::new();
    declared.try_reserve_exact(tensor_infos.len()).map_err(|_| {
        GGUFError::InvalidTensorData("Unable to allocate declared tensor state".to_string())
    })?;
    let mut names = HashSet::new();
    names.try_reserve(tensor_infos.len()).map_err(|_| {
        GGUFError::InvalidTensorData("Unable to allocate tensor-name set".to_string())
    })?;
    let mut previous_end = 0u64;

    for tensor_info in tensor_infos {
        tensor_info.validate()?;
        if !tensor_info.data_offset().is_multiple_of(alignment) {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' offset {} is not aligned to {} bytes",
                tensor_info.name(),
                tensor_info.data_offset(),
                alignment
            )));
        }
        if !names.insert(tensor_info.name()) {
            return Err(GGUFError::InvalidTensorData(format!(
                "Duplicate tensor name: {}",
                tensor_info.name()
            )));
        }
        let expected_size_u64 = tensor_info.checked_expected_data_size()?;
        let expected_size = usize::try_from(expected_size_u64).map_err(|_| {
            GGUFError::InvalidTensorData(format!(
                "Tensor '{}' size does not fit this platform",
                tensor_info.name()
            ))
        })?;
        if tensor_info.data_offset() < previous_end {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' offset is out of order or overlaps a previous tensor",
                tensor_info.name()
            )));
        }
        previous_end =
            tensor_info.data_offset().checked_add(expected_size_u64).ok_or_else(|| {
                GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' range overflows u64",
                    tensor_info.name()
                ))
            })?;
        declared
            .push(DeclaredTensor { descriptor: format_tensor_info(tensor_info)?, expected_size });
    }
    Ok(declared)
}

fn format_tensor_info(tensor_info: &TensorInfoNew) -> Result<TensorInfo> {
    let name = try_clone_string(tensor_info.name(), "tensor name")?;
    let mut dimensions = Vec::new();
    dimensions.try_reserve_exact(tensor_info.shape().dims().len()).map_err(|_| {
        GGUFError::InvalidTensorData("Unable to allocate tensor dimensions".to_string())
    })?;
    dimensions.extend_from_slice(tensor_info.shape().dims());
    Ok(TensorInfo::new(
        name,
        dimensions,
        tensor_info.tensor_type() as u32,
        tensor_info.data_offset(),
    ))
}

fn clone_tensor_info_with_offset(
    tensor_info: &TensorInfoNew,
    offset: u64,
) -> Result<TensorInfoNew> {
    let name = try_clone_string(tensor_info.name(), "tensor name")?;
    let mut dimensions = Vec::new();
    dimensions.try_reserve_exact(tensor_info.shape().dims().len()).map_err(|_| {
        GGUFError::InvalidTensorData("Unable to allocate tensor dimensions".to_string())
    })?;
    dimensions.extend_from_slice(tensor_info.shape().dims());
    let shape = TensorShape::new(dimensions)?;
    Ok(TensorInfoNew::new(name, shape, tensor_info.tensor_type(), offset))
}

fn try_clone_string(value: &str, description: &str) -> Result<String> {
    let mut cloned = String::new();
    cloned
        .try_reserve_exact(value.len())
        .map_err(|_| GGUFError::InvalidTensorData(format!("Unable to allocate {description}")))?;
    cloned.push_str(value);
    Ok(cloned)
}

/// Convenience function to create a GGUF file at a path
pub fn create_gguf_file<P: AsRef<Path>>(
    path: P,
    metadata: &Metadata,
    tensors: &[(TensorInfoNew, TensorData)],
) -> Result<GGUFWriteResult> {
    create_gguf_file_with_config(path, metadata, tensors, GGUFWriterConfig::default())
}

/// Convenience function to create a GGUF file with custom configuration
pub fn create_gguf_file_with_config<P: AsRef<Path>>(
    path: P,
    metadata: &Metadata,
    tensors: &[(TensorInfoNew, TensorData)],
    config: GGUFWriterConfig,
) -> Result<GGUFWriteResult> {
    // Preflight before opening the path so invalid caller input cannot truncate
    // an existing destination. I/O failures after this point can still leave a
    // partial file and must be retried against a fresh target.
    let preflight = GGUFFileWriter::with_config(std::io::sink(), config.clone());
    let prepared = preflight.prepare_complete_file(metadata, tensors)?;

    let file = File::create(path)?;
    let buf_writer = BufWriter::new(file);
    let mut writer = GGUFFileWriter::with_config(buf_writer, config);

    writer.write_prepared_complete_file(prepared, tensors)
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
    use crate::reader::GGUFFileReader;
    use crate::tensor::{TensorShape, TensorType};
    use std::io::{Cursor, Error, ErrorKind};

    #[derive(Default)]
    struct FlushFailWriter(Vec<u8>);

    impl Write for FlushFailWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Err(Error::other("deliberate flush failure"))
        }
    }

    fn create_test_metadata() -> Metadata {
        let mut metadata = Metadata::new();
        metadata.insert("name".to_string(), MetadataValue::String("test_model".to_string()));
        metadata.insert("version".to_string(), MetadataValue::U32(1));
        metadata
    }

    #[test]
    fn complete_file_propagates_flush_errors() {
        let mut writer = GGUFFileWriter::new(FlushFailWriter::default());
        let error = writer.write_complete_file(&Metadata::new(), &[]).unwrap_err();
        assert!(matches!(error, GGUFError::Io(error) if error.kind() == ErrorKind::Other));
    }

    #[test]
    fn complete_file_validates_all_input_before_writing() {
        let info = TensorInfoNew::new(
            "bad".to_string(),
            TensorShape::new(vec![1]).unwrap(),
            TensorType::F32,
            0,
        );
        let mut bytes = Vec::new();
        let result = {
            let mut writer = GGUFFileWriter::new(&mut bytes);
            writer.write_complete_file(&Metadata::new(), &[(info, TensorData::new_owned(vec![0]))])
        };
        assert!(result.is_err());
        assert!(bytes.is_empty());
    }

    fn create_test_tensor() -> (TensorInfoNew, TensorData) {
        let shape = TensorShape::new(vec![2, 2]).unwrap();
        let tensor_info = TensorInfoNew::new("test_tensor".to_string(), shape, TensorType::F32, 0);
        let data = TensorData::new_owned(vec![0u8; 16]); // 4 F32 values
        (tensor_info, data)
    }

    fn create_ordered_test_tensors() -> Vec<(TensorInfoNew, TensorData)> {
        vec![
            (
                TensorInfoNew::new(
                    "first".to_string(),
                    TensorShape::new(vec![1]).unwrap(),
                    TensorType::F32,
                    0,
                ),
                TensorData::new_owned(vec![1, 2, 3, 4]),
            ),
            (
                TensorInfoNew::new(
                    "second".to_string(),
                    TensorShape::new(vec![1]).unwrap(),
                    TensorType::F32,
                    32,
                ),
                TensorData::new_owned(vec![5, 6, 7, 8]),
            ),
        ]
    }

    fn prepare_low_level_writer(
        writer: &mut GGUFFileWriter<Vec<u8>>,
        tensors: &[(TensorInfoNew, TensorData)],
    ) {
        writer.write_header(&GGUFHeader::new(tensors.len() as u64, 0)).unwrap();
        writer.write_metadata(&Metadata::new()).unwrap();
        let infos: Vec<_> = tensors.iter().map(|(info, _)| info.clone()).collect();
        writer.write_tensor_infos(&infos).unwrap();
        writer.align_for_tensor_data().unwrap();
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

        let unsupported = GGUFWriterConfig { compress_metadata: true, ..Default::default() };
        let mut writer = GGUFFileWriter::with_config(Vec::new(), unsupported);
        assert!(matches!(
            writer.write_header(&GGUFHeader::default()),
            Err(GGUFError::FeatureUnavailable(_))
        ));
    }

    #[test]
    fn test_custom_alignment_is_declared_and_round_trips() {
        let config = GGUFWriterConfig { tensor_alignment: 64, ..Default::default() };
        let mut writer = GGUFFileWriter::with_config(Vec::new(), config);
        let (info, data) = create_test_tensor();
        writer.write_complete_file(&Metadata::new(), &[(info, data)]).unwrap();
        let bytes = writer.finalize().unwrap();

        let reader = GGUFFileReader::new(Cursor::new(bytes)).unwrap();
        assert_eq!(reader.tensor_alignment(), 64);
        assert!(reader.tensor_data_offset().is_multiple_of(64));
        assert_eq!(reader.metadata().get_u64("general.alignment"), Some(64));
    }

    #[test]
    fn test_write_header() {
        let buffer = Vec::new();
        let mut writer = GGUFFileWriter::new(buffer);

        let header = GGUFHeader::new(1, 2);
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
        let header = GGUFHeader::new(1, 2);
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
        let header = GGUFHeader::new(1, 2);
        writer.write_header(&header).unwrap();

        // Now metadata should work
        let result = writer.write_metadata(&metadata);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tensor_data_validation() {
        let buffer = Vec::new();
        let mut writer = GGUFFileWriter::new(buffer);

        let header = GGUFHeader::new(1, 2);
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
    fn test_payload_descriptors_must_follow_declared_order_once() {
        let tensors = create_ordered_test_tensors();
        let mut writer = GGUFFileWriter::new(Vec::new());
        prepare_low_level_writer(&mut writer, &tensors);

        assert!(writer.write_tensor_data(&tensors[1].0, &tensors[1].1).is_err());
        writer.write_tensor_data(&tensors[0].0, &tensors[0].1).unwrap();
        assert!(writer.write_tensor_data(&tensors[0].0, &tensors[0].1).is_err());
        writer.write_tensor_data(&tensors[1].0, &tensors[1].1).unwrap();
        assert!(writer.finalize().is_ok());
    }

    #[test]
    fn test_finalize_rejects_missing_tensor_payloads() {
        let tensors = create_ordered_test_tensors();
        let mut writer = GGUFFileWriter::new(Vec::new());
        prepare_low_level_writer(&mut writer, &tensors);
        writer.write_tensor_data(&tensors[0].0, &tensors[0].1).unwrap();

        assert!(matches!(writer.finalize(), Err(GGUFError::InvalidTensorData(_))));
    }

    #[test]
    fn test_out_of_order_descriptor_offsets_are_rejected() {
        let mut tensors = create_ordered_test_tensors();
        tensors.swap(0, 1);
        let mut writer = GGUFFileWriter::new(Vec::new());
        writer.write_header(&GGUFHeader::new(2, 0)).unwrap();
        writer.write_metadata(&Metadata::new()).unwrap();
        let infos: Vec<_> = tensors.iter().map(|(info, _)| info.clone()).collect();

        assert!(matches!(
            writer.write_tensor_infos(&infos),
            Err(GGUFError::InvalidTensorData(message)) if message.contains("out of order")
        ));
    }

    #[test]
    fn test_alignment() {
        let buffer = Vec::new();
        let mut writer = GGUFFileWriter::new(buffer);

        let header = GGUFHeader::new(1, 2);
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
    fn path_helper_does_not_truncate_destination_when_preflight_fails() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("existing.gguf");
        let original = b"existing destination";
        std::fs::write(&path, original).unwrap();

        let invalid_tensor = TensorInfoNew::new(
            "bad".to_string(),
            TensorShape::new(vec![1]).unwrap(),
            TensorType::F32,
            0,
        );
        let result = create_gguf_file(
            &path,
            &Metadata::new(),
            &[(invalid_tensor, TensorData::new_owned(vec![0]))],
        );

        assert!(matches!(result, Err(GGUFError::InvalidTensorData(_))));
        assert_eq!(std::fs::read(path).unwrap(), original);
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
