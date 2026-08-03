//! Stream-based GGUF writer for non-seekable streams

use crate::error::{GGUFError, Result};
use crate::format::{
    alignment::{AlignmentInfo, AlignmentTracker},
    constants::GGUF_DEFAULT_ALIGNMENT,
    GGUFHeader, Metadata, MetadataValue, TensorInfo,
};
use crate::tensor::{TensorData, TensorInfo as TensorInfoNew, TensorShape};
use std::collections::HashSet;
use std::io::Write;

const MAX_WRITE_CHUNK_SIZE: usize = 1024 * 1024;

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
    /// Tensor count declared by the header
    expected_tensor_count: u64,
    /// Metadata count declared by the header
    expected_metadata_count: u64,
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

impl DeclaredTensor {
    fn matches(&self, tensor_info: &TensorInfoNew) -> bool {
        self.descriptor.name == tensor_info.name()
            && self.descriptor.dimensions == tensor_info.shape().dims()
            && self.descriptor.tensor_type == tensor_info.tensor_type() as u32
            && self.descriptor.offset == tensor_info.data_offset()
    }
}

/// Configuration for stream writing
#[derive(Debug, Clone)]
pub struct StreamWriterConfig {
    /// Tensor data alignment
    pub tensor_alignment: usize,
    /// Request optional content validation beyond mandatory format invariants.
    ///
    /// Descriptor validity and exact payload lengths are always enforced.
    pub validate_data: bool,
    /// Preferred chunk size; temporary chunks are capped at 1 MiB.
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
            expected_tensor_count: 0,
            expected_metadata_count: 0,
            declared_tensors: Vec::new(),
            next_tensor: 0,
            tensor_data_start: None,
        }
    }

    /// Write the header
    pub fn write_header(&mut self, header: &GGUFHeader) -> Result<StreamWriteResult> {
        if self.state != WriterState::Ready {
            return Err(GGUFError::Format("Header already written or invalid state".to_string()));
        }

        self.validate_config()?;
        header.validate_comprehensive()?;

        header.write_to(&mut self.writer)?;

        let bytes_written = GGUFHeader::size();
        self.advance(bytes_written)?;
        self.state = WriterState::HeaderWritten;
        self.expected_tensor_count = header.tensor_count;
        self.expected_metadata_count = header.metadata_kv_count;

        Ok(StreamWriteResult { bytes_written, current_position: self.position, validated: true })
    }

    /// Write metadata
    pub fn write_metadata(&mut self, metadata: &Metadata) -> Result<StreamWriteResult> {
        if self.state != WriterState::HeaderWritten {
            return Err(GGUFError::Format("Must write header before metadata".to_string()));
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
        }

        self.advance(total_bytes)?;
        self.declared_tensors = declared_tensors;
        self.next_tensor = 0;
        self.state = WriterState::TensorInfoWritten;

        Ok(StreamWriteResult {
            bytes_written: total_bytes,
            current_position: self.position,
            validated: true,
        })
    }

    /// Align for tensor data section
    pub fn align_for_tensor_data(&mut self) -> Result<StreamWriteResult> {
        if self.state != WriterState::TensorInfoWritten {
            return Err(GGUFError::Format("Must write tensor info before alignment".to_string()));
        }

        self.validate_config()?;
        let alignment_info =
            AlignmentInfo::new(self.alignment_tracker.position, self.config.tensor_alignment);

        let bytes_written = if alignment_info.needs_padding() {
            self.write_padding(alignment_info.padding)?;
            alignment_info.padding
        } else {
            0
        };

        self.state = WriterState::TensorDataReady;
        self.tensor_data_start = Some(self.position);

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

        let (expected_offset, expected_size) = self.expected_tensor(tensor_info)?;
        if data.len() != expected_size {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' size mismatch: expected {}, got {}",
                tensor_info.name(),
                expected_size,
                data.len()
            )));
        }

        let data_bytes = data.try_as_slice()?;
        self.pad_to_tensor_offset(expected_offset, tensor_info.name())?;
        self.writer.write_all(data_bytes)?;

        let bytes_written = data_bytes.len();
        self.advance(data_bytes.len())?;
        self.mark_tensor_written()?;
        self.state = WriterState::WritingTensors;

        Ok(StreamWriteResult { bytes_written, current_position: self.position, validated: true })
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

        let (expected_offset, expected_size) = self.expected_tensor(tensor_info)?;
        if expected_size > 0 && self.config.buffer_size == 0 {
            return Err(GGUFError::InvalidTensorData(
                "Stream writer buffer size must be greater than zero".to_string(),
            ));
        }
        self.pad_to_tensor_offset(expected_offset, tensor_info.name())?;
        let chunk_size = self.config.buffer_size.min(expected_size).min(MAX_WRITE_CHUNK_SIZE);
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(chunk_size).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor write chunk".to_string())
        })?;
        buffer.resize(chunk_size, 0);
        let mut total_written = 0;

        while total_written < expected_size {
            let to_read = (expected_size - total_written).min(buffer.len());
            reader.read_exact(&mut buffer[..to_read])?;
            self.writer.write_all(&buffer[..to_read])?;

            total_written = total_written.checked_add(to_read).ok_or_else(|| {
                GGUFError::InvalidTensorData("Tensor write size overflows usize".to_string())
            })?;
        }

        self.advance(total_written)?;
        self.mark_tensor_written()?;
        self.state = WriterState::WritingTensors;

        Ok(StreamWriteResult {
            bytes_written: total_written,
            current_position: self.position,
            validated: true,
        })
    }

    /// Write a complete GGUF file to stream
    pub fn write_complete_stream(
        &mut self,
        metadata: &Metadata,
        tensors: &[(TensorInfoNew, TensorData)],
    ) -> Result<CompleteStreamWriteResult> {
        self.validate_config()?;
        let metadata = self.metadata_with_declared_alignment(metadata)?;
        let header = GGUFHeader::new(
            u64::try_from(tensors.len())
                .map_err(|_| GGUFError::Format("Tensor count does not fit u64".to_string()))?,
            u64::try_from(metadata.len())
                .map_err(|_| GGUFError::Format("Metadata count does not fit u64".to_string()))?,
        );

        // Validate the complete caller-owned model before writing any bytes.
        // I/O failures can still leave partial output, but invalid descriptors,
        // duplicate names, and bad payloads fail without corrupting the target.
        let mut tensor_infos = Vec::new();
        tensor_infos.try_reserve_exact(tensors.len()).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor descriptor list".to_string())
        })?;
        let mut tensor_names = HashSet::new();
        tensor_names.try_reserve(tensors.len()).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor-name set".to_string())
        })?;
        let mut current_offset = 0u64;
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
            current_offset =
                checked_align_u64(current_offset, self.config.tensor_alignment as u64)?;
            tensor_infos.push(clone_tensor_info_with_offset(tensor_info, current_offset)?);
            current_offset = current_offset.checked_add(expected_size).ok_or_else(|| {
                GGUFError::InvalidTensorData("Tensor data offsets overflow u64".to_string())
            })?;
        }

        // Write only after all format-level input validation has succeeded.
        let header_result = self.write_header(&header)?;
        let metadata_result = self.write_metadata(&metadata)?;
        let tensor_info_result = self.write_tensor_infos(&tensor_infos)?;

        // Align for tensor data
        let alignment_result = self.align_for_tensor_data()?;

        // Write tensor data
        let mut tensor_results = Vec::new();
        tensor_results.try_reserve_exact(tensors.len()).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor write results".to_string())
        })?;
        for ((_, tensor_data), tensor_info) in tensors.iter().zip(tensor_infos.iter()) {
            let result = self.write_tensor_data(tensor_info, tensor_data)?;
            tensor_results.push(result);
        }

        self.ensure_all_tensor_payloads_written()?;
        self.writer.flush()?;
        self.state = WriterState::Finished;

        let total_bytes = usize::try_from(self.position).map_err(|_| {
            GGUFError::Format("Final writer position does not fit usize".to_string())
        })?;

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
        if self.state != WriterState::Finished {
            self.ensure_all_tensor_payloads_written()?;
        }
        self.writer.flush()?;
        self.state = WriterState::Finished;
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

    fn validate_config(&self) -> Result<()> {
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

    fn expected_tensor(&self, tensor_info: &TensorInfoNew) -> Result<(u64, usize)> {
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
        Ok((expected.descriptor.offset, expected.expected_size))
    }

    fn pad_to_tensor_offset(&mut self, offset: u64, name: &str) -> Result<()> {
        let data_start = self.tensor_data_start.ok_or_else(|| {
            GGUFError::Format("Tensor data section start is unavailable".to_string())
        })?;
        let target = data_start.checked_add(offset).ok_or_else(|| {
            GGUFError::InvalidTensorData("Tensor absolute offset overflows u64".to_string())
        })?;
        if self.position > target {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' offset precedes the current write position",
                name
            )));
        }
        let padding = usize::try_from(target - self.position).map_err(|_| {
            GGUFError::InvalidTensorData("Tensor padding does not fit usize".to_string())
        })?;
        self.write_padding(padding)
    }

    fn mark_tensor_written(&mut self) -> Result<()> {
        self.next_tensor = self.next_tensor.checked_add(1).ok_or_else(|| {
            GGUFError::InvalidTensorData("Tensor payload cursor overflows usize".to_string())
        })?;
        Ok(())
    }

    fn ensure_all_tensor_payloads_written(&self) -> Result<()> {
        if !matches!(self.state, WriterState::TensorDataReady | WriterState::WritingTensors) {
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
        self.tensor_results
            .iter()
            .fold(0usize, |total, result| total.saturating_add(result.bytes_written))
    }

    /// Get overhead bytes
    pub fn overhead_bytes(&self) -> usize {
        self.total_bytes_written.saturating_sub(self.tensor_data_bytes())
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

    fn create_test_setup() -> (Metadata, Vec<(TensorInfoNew, TensorData)>) {
        let mut metadata = Metadata::new();
        metadata.insert("name".to_string(), MetadataValue::String("test".to_string()));

        let shape = TensorShape::new(vec![2, 2]).unwrap();
        let tensor_info = TensorInfoNew::new("tensor".to_string(), shape, TensorType::F32, 0);
        let data = TensorData::new_owned(vec![0u8; 16]);

        (metadata, vec![(tensor_info, data)])
    }

    #[test]
    fn complete_stream_propagates_flush_errors() {
        let mut writer = GGUFStreamWriter::new(FlushFailWriter::default());
        let error = writer.write_complete_stream(&Metadata::new(), &[]).unwrap_err();
        assert!(matches!(error, GGUFError::Io(error) if error.kind() == ErrorKind::Other));
    }

    #[test]
    fn complete_stream_validates_all_input_before_writing() {
        let info = TensorInfoNew::new(
            "bad".to_string(),
            TensorShape::new(vec![1]).unwrap(),
            TensorType::F32,
            0,
        );
        let mut bytes = Vec::new();
        let result = {
            let mut writer = GGUFStreamWriter::new(&mut bytes);
            writer
                .write_complete_stream(&Metadata::new(), &[(info, TensorData::new_owned(vec![0]))])
        };
        assert!(result.is_err());
        assert!(bytes.is_empty());
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
        writer: &mut GGUFStreamWriter<Vec<u8>>,
        tensors: &[(TensorInfoNew, TensorData)],
    ) {
        writer.write_header(&GGUFHeader::new(tensors.len() as u64, 0)).unwrap();
        writer.write_metadata(&Metadata::new()).unwrap();
        let infos: Vec<_> = tensors.iter().map(|(info, _)| info.clone()).collect();
        writer.write_tensor_infos(&infos).unwrap();
        writer.align_for_tensor_data().unwrap();
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
    fn test_two_small_tensors_round_trip_with_alignment() {
        let metadata = Metadata::new();
        let tensors = vec![
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
                    0,
                ),
                TensorData::new_owned(vec![5, 6, 7, 8]),
            ),
        ];

        let mut writer = GGUFStreamWriter::new(Vec::new());
        let result = writer.write_complete_stream(&metadata, &tensors).unwrap();
        assert_eq!(result.tensor_data_bytes(), 8);
        assert_eq!(result.overhead_bytes() + 8, result.total_bytes_written);

        let bytes = writer.into_inner();
        let mut reader = GGUFFileReader::new(Cursor::new(bytes)).unwrap();
        assert_eq!(reader.get_tensor_info("first").unwrap().data_offset(), 0);
        assert_eq!(reader.get_tensor_info("second").unwrap().data_offset(), 32);
        assert_eq!(reader.load_tensor_data("first").unwrap().unwrap().as_slice(), &[1, 2, 3, 4]);
        assert_eq!(reader.load_tensor_data("second").unwrap().unwrap().as_slice(), &[5, 6, 7, 8]);
    }

    #[test]
    fn test_payload_size_is_always_enforced() {
        let config = StreamWriterConfig { validate_data: false, ..Default::default() };
        let mut writer = GGUFStreamWriter::with_config(Vec::new(), config);
        writer.write_header(&GGUFHeader::new(1, 0)).unwrap();
        writer.write_metadata(&Metadata::new()).unwrap();
        let info = TensorInfoNew::new(
            "tensor".to_string(),
            TensorShape::new(vec![2]).unwrap(),
            TensorType::F32,
            0,
        );
        writer.write_tensor_infos(std::slice::from_ref(&info)).unwrap();
        writer.align_for_tensor_data().unwrap();
        assert!(writer.write_tensor_data(&info, &TensorData::new_owned(vec![0; 4])).is_err());
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
        writer.write_tensor_infos(std::slice::from_ref(&tensor_info)).unwrap();
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
    fn test_payload_descriptors_must_follow_declared_order_once() {
        let tensors = create_ordered_test_tensors();
        let mut writer = GGUFStreamWriter::new(Vec::new());
        prepare_low_level_writer(&mut writer, &tensors);

        assert!(writer.write_tensor_data(&tensors[1].0, &tensors[1].1).is_err());
        writer.write_tensor_data(&tensors[0].0, &tensors[0].1).unwrap();
        assert!(writer.write_tensor_data(&tensors[0].0, &tensors[0].1).is_err());
        writer.write_tensor_data(&tensors[1].0, &tensors[1].1).unwrap();
        writer.finalize().unwrap();
        assert!(writer.is_finished());
    }

    #[test]
    fn test_finalize_rejects_missing_tensor_payloads() {
        let tensors = create_ordered_test_tensors();
        let mut writer = GGUFStreamWriter::new(Vec::new());
        prepare_low_level_writer(&mut writer, &tensors);
        writer.write_tensor_data(&tensors[0].0, &tensors[0].1).unwrap();

        assert!(matches!(writer.finalize(), Err(GGUFError::InvalidTensorData(_))));
        assert!(!writer.is_finished());
        writer.write_tensor_data(&tensors[1].0, &tensors[1].1).unwrap();
        writer.finalize().unwrap();
    }

    #[test]
    fn test_out_of_order_descriptor_offsets_are_rejected() {
        let mut tensors = create_ordered_test_tensors();
        tensors.swap(0, 1);
        let mut writer = GGUFStreamWriter::new(Vec::new());
        writer.write_header(&GGUFHeader::new(2, 0)).unwrap();
        writer.write_metadata(&Metadata::new()).unwrap();
        let infos: Vec<_> = tensors.iter().map(|(info, _)| info.clone()).collect();

        assert!(matches!(
            writer.write_tensor_infos(&infos),
            Err(GGUFError::InvalidTensorData(message)) if message.contains("out of order")
        ));
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
        writer.write_tensor_infos(std::slice::from_ref(&tensor_info)).unwrap();
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
