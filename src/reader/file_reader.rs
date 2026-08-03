//! File-based GGUF reader
//!
//! This module provides functionality for reading GGUF files from various sources.
//!
//! ## Example
//!
//! ```rust
//! # use gguf_rs_lib::prelude::*;
//! # use std::io::Cursor;
//! # fn example_data() -> Vec<u8> {
//! #     use gguf_rs_lib::format::constants::*;
//! #     let mut data = Vec::new();
//! #     // Header
//! #     data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
//! #     data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
//! #     data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
//! #     data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata entry
//! #     // Metadata
//! #     data.extend_from_slice(&4u64.to_le_bytes()); // key length
//! #     data.extend_from_slice(b"name"); // key
//! #     data.extend_from_slice(&8u32.to_le_bytes()); // string type
//! #     data.extend_from_slice(&5u64.to_le_bytes()); // value length
//! #     data.extend_from_slice(b"model"); // value
//! #     while data.len() % 32 != 0 { data.push(0); } // alignment
//! #     data
//! # }
//! # fn main() -> Result<()> {
//! let data = example_data();
//! let mut reader = GGUFFileReader::new(Cursor::new(data))?;
//!
//! // Access file information
//! println!("GGUF version: {}", reader.header().version);
//! println!("Tensors: {}", reader.tensor_count());
//!
//! // Get a summary
//! let summary = reader.summary();
//! println!("Summary: {}", summary);
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "std")]
use crate::error::{GGUFError, Result};
#[cfg(feature = "std")]
use crate::format::types::GGUFTensorType as TensorType;
#[cfg(feature = "std")]
use crate::format::{
    constants::{GGUF_MAX_METADATA_DECODED_SIZE, GGUF_MAX_METADATA_SIZE},
    GGUFHeader, Metadata, TensorInfo,
};
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

const MIN_TENSOR_TRACKING_CAPACITY: usize = 64;
const MAX_READER_CHUNK_SIZE: usize = 1024 * 1024;

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
    /// Tensor descriptor index by name
    tensor_name_index: HashMap<String, usize>,
    /// Current position in the file
    position: u64,
    /// Start of tensor data section
    tensor_data_offset: u64,
    /// Alignment required for every tensor offset
    tensor_alignment: u64,
    /// Total input length captured at construction time
    file_size: u64,
    /// Maximum temporary buffer used while growing owned tensor data
    buffer_size: usize,
}

/// Configuration for GGUF file reading
#[derive(Debug, Clone)]
pub struct GGUFReaderConfig {
    /// Whether to validate descriptor consistency and payload ranges.
    ///
    /// This does not read payload bytes or verify a cryptographic checksum.
    pub validate_integrity: bool,
    /// Whether to load tensor data immediately
    pub eager_load_tensors: bool,
    /// Maximum file size to read (0 = no limit)
    pub max_file_size: u64,
    /// Maximum number of serialized metadata bytes to read.
    pub max_metadata_size: usize,
    /// Maximum decoded allocation budget for metadata.
    pub max_decoded_metadata_size: usize,
    /// Preferred buffer size for reading; temporary chunks are capped at 1 MiB.
    pub buffer_size: usize,
    /// Request automatic memory mapping.
    ///
    /// Automatic mapping is not supported by the generic reader; setting this to
    /// `true` returns [`GGUFError::FeatureUnavailable`].
    pub use_mmap: bool,
}

impl Default for GGUFReaderConfig {
    fn default() -> Self {
        Self {
            validate_integrity: true,
            eager_load_tensors: false,
            max_file_size: 0,
            max_metadata_size: GGUF_MAX_METADATA_SIZE,
            max_decoded_metadata_size: GGUF_MAX_METADATA_DECODED_SIZE,
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
        if config.use_mmap {
            return Err(GGUFError::FeatureUnavailable(
                "automatic memory mapping in GGUFFileReader".to_string(),
            ));
        }
        // Read and validate header
        let header = GGUFHeader::read_from(&mut reader)?;
        header.validate_comprehensive()?;

        // Capture the file size once so descriptor ranges can be validated before allocation.
        let current_pos = reader.stream_position()?;
        let file_size = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(current_pos))?;
        if config.max_file_size > 0 && file_size > config.max_file_size {
            return Err(GGUFError::Format(format!(
                "File size {} exceeds maximum allowed size {}",
                file_size, config.max_file_size
            )));
        }

        // Read metadata
        let metadata = Metadata::read_from_with_limits(
            &mut reader,
            header.metadata_kv_count,
            config.max_metadata_size,
            config.max_decoded_metadata_size,
        )?;
        let tensor_alignment = u64::try_from(metadata.tensor_alignment()?).map_err(|_| {
            GGUFError::InvalidMetadata("Tensor alignment does not fit u64".to_string())
        })?;

        // Read tensor information
        let tensor_capacity = usize::try_from(header.tensor_count).map_err(|_| {
            GGUFError::InvalidTensorData("Tensor count does not fit this platform".to_string())
        })?;
        let mut tensor_infos = Vec::new();
        for _ in 0..header.tensor_count {
            let tensor_info = TensorInfo::read_from(&mut reader)?;

            // Convert to our TensorInfo format
            let shape = TensorShape::new(tensor_info.dimensions)?;
            let tensor_type = TensorType::from_u32(tensor_info.tensor_type)?;

            let new_tensor_info =
                TensorInfoNew::new(tensor_info.name, shape, tensor_type, tensor_info.offset);
            new_tensor_info.validate()?;
            if !new_tensor_info.data_offset().is_multiple_of(tensor_alignment) {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' offset {} is not aligned to {} bytes",
                    new_tensor_info.name(),
                    new_tensor_info.data_offset(),
                    tensor_alignment
                )));
            }
            try_reserve_vec_slot(&mut tensor_infos, tensor_capacity, "tensor descriptor list")?;
            tensor_infos.push(new_tensor_info);
        }

        let mut tensor_name_index = HashMap::new();
        for (index, tensor_info) in tensor_infos.iter().enumerate() {
            try_reserve_map_slot(&mut tensor_name_index, tensor_infos.len(), "tensor name index")?;
            let name = try_clone_tensor_name(tensor_info.name())?;
            if tensor_name_index.insert(name, index).is_some() {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Duplicate tensor name: {}",
                    tensor_info.name()
                )));
            }
        }

        // Calculate tensor data section offset
        let current_position = reader.stream_position()?;
        let tensor_data_offset = checked_align_u64(current_position, tensor_alignment)?;
        if header.tensor_count > 0 && tensor_data_offset > file_size {
            return Err(GGUFError::UnexpectedEof);
        }

        let mut gguf_reader = Self {
            reader,
            header,
            metadata,
            tensor_infos,
            tensor_name_index,
            position: current_position,
            tensor_data_offset,
            tensor_alignment,
            file_size,
            buffer_size: config.buffer_size,
        };

        // Validate descriptors before eager payload allocation so malformed
        // ranges and overlaps fail without reading tensor data.
        if config.validate_integrity {
            gguf_reader.validate_integrity()?;
        }

        // Eager load tensor data if requested
        if config.eager_load_tensors {
            gguf_reader.load_all_tensor_data()?;
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
        self.tensor_name_index.get(name).and_then(|&index| self.tensor_infos.get(index))
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
        let tensor_index =
            self.tensor_name_index.get(name).copied().ok_or_else(|| {
                GGUFError::InvalidTensorData(format!("Tensor '{}' not found", name))
            })?;

        self.load_tensor_data_by_index(tensor_index).map(Some)
    }

    fn load_tensor_data_by_index(&mut self, tensor_index: usize) -> Result<TensorData> {
        let tensor_info = self.tensor_infos.get(tensor_index).ok_or_else(|| {
            GGUFError::InvalidTensorData("Tensor descriptor index is out of bounds".to_string())
        })?;
        let data_size_u64 = tensor_info.checked_expected_data_size()?;
        let data_size = usize::try_from(data_size_u64).map_err(|_| {
            GGUFError::InvalidTensorData(format!(
                "Tensor '{}' size does not fit this platform",
                tensor_info.name()
            ))
        })?;

        // Seek to tensor data
        let absolute_offset =
            self.tensor_data_offset.checked_add(tensor_info.data_offset()).ok_or_else(|| {
                GGUFError::InvalidTensorData("Tensor offset overflows u64".to_string())
            })?;
        let absolute_end = absolute_offset.checked_add(data_size_u64).ok_or_else(|| {
            GGUFError::InvalidTensorData("Tensor range overflows u64".to_string())
        })?;
        if absolute_end > self.file_size {
            return Err(GGUFError::UnexpectedEof);
        }
        self.reader.seek(SeekFrom::Start(absolute_offset))?;

        let data = match read_exact_owned(&mut self.reader, data_size, self.buffer_size) {
            Ok(data) => data,
            Err(error) => {
                if let Ok(position) = self.reader.stream_position() {
                    self.position = position;
                }
                return Err(error);
            }
        };
        self.position = absolute_end;

        Ok(TensorData::new_owned(data))
    }

    /// Load and retain every tensor payload.
    ///
    /// This can require memory comparable to the complete model. Use
    /// [`Self::validate_all_tensor_data`] to check readability with bounded
    /// temporary memory instead.
    pub fn load_all_tensor_data(&mut self) -> Result<()> {
        for tensor_index in 0..self.tensor_infos.len() {
            let tensor_data = self.load_tensor_data_by_index(tensor_index)?;
            self.tensor_infos[tensor_index].set_data(tensor_data);
        }

        Ok(())
    }

    /// Read and discard every declared tensor payload.
    ///
    /// This verifies that every descriptor range is readable without retaining
    /// tensor data. A single bounded, fallibly allocated buffer is reused for
    /// the whole file.
    pub fn validate_all_tensor_data(&mut self) -> Result<()> {
        if self.tensor_infos.is_empty() {
            return Ok(());
        }
        let max_payload_size = self.tensor_infos.iter().try_fold(0usize, |maximum, tensor| {
            let size = usize::try_from(tensor.checked_expected_data_size()?).map_err(|_| {
                GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' size does not fit this platform",
                    tensor.name()
                ))
            })?;
            Ok::<_, GGUFError>(maximum.max(size))
        })?;
        if max_payload_size > 0 && self.buffer_size == 0 {
            return Err(GGUFError::InvalidTensorData(
                "File reader buffer size must be greater than zero".to_string(),
            ));
        }
        let chunk_size = self.buffer_size.min(MAX_READER_CHUNK_SIZE).min(max_payload_size);
        let mut buffer = try_zeroed_buffer(chunk_size, "tensor validation chunk")?;

        for tensor_index in 0..self.tensor_infos.len() {
            let (absolute_offset, data_size) = self.tensor_range_by_index(tensor_index)?;
            self.reader.seek(SeekFrom::Start(absolute_offset))?;
            self.position = absolute_offset;
            let mut remaining = data_size;
            while remaining > 0 {
                let to_read = remaining.min(buffer.len());
                self.read_exact_tracking(&mut buffer[..to_read])?;
                remaining -= to_read;
            }
        }
        Ok(())
    }

    /// Compare a named tensor payload with another seekable GGUF reader.
    ///
    /// Comparison uses two bounded buffers and does not retain either payload.
    pub fn tensor_data_equals<S: Read + Seek>(
        &mut self,
        name: &str,
        other: &mut GGUFFileReader<S>,
    ) -> Result<bool> {
        let tensor_index =
            self.tensor_name_index.get(name).copied().ok_or_else(|| {
                GGUFError::InvalidTensorData(format!("Tensor '{}' not found", name))
            })?;
        let other_index =
            other.tensor_name_index.get(name).copied().ok_or_else(|| {
                GGUFError::InvalidTensorData(format!("Tensor '{}' not found", name))
            })?;
        let (absolute_offset, data_size) = self.tensor_range_by_index(tensor_index)?;
        let (other_offset, other_size) = other.tensor_range_by_index(other_index)?;
        if data_size != other_size {
            return Ok(false);
        }
        if data_size == 0 {
            return Ok(true);
        }
        if self.buffer_size == 0 || other.buffer_size == 0 {
            return Err(GGUFError::InvalidTensorData(
                "File reader buffer size must be greater than zero".to_string(),
            ));
        }
        let chunk_size = self
            .buffer_size
            .min(other.buffer_size)
            .min(MAX_READER_CHUNK_SIZE)
            .min(data_size);
        let mut left = try_zeroed_buffer(chunk_size, "tensor comparison chunk")?;
        let mut right = try_zeroed_buffer(chunk_size, "tensor comparison chunk")?;
        self.reader.seek(SeekFrom::Start(absolute_offset))?;
        self.position = absolute_offset;
        other.reader.seek(SeekFrom::Start(other_offset))?;
        other.position = other_offset;

        let mut remaining = data_size;
        while remaining > 0 {
            let to_read = remaining.min(chunk_size);
            self.read_exact_tracking(&mut left[..to_read])?;
            other.read_exact_tracking(&mut right[..to_read])?;
            if left[..to_read] != right[..to_read] {
                return Ok(false);
            }
            remaining -= to_read;
        }
        Ok(true)
    }

    /// Read tensor data at a specific offset and size
    pub fn read_tensor_data_at(&mut self, offset: u64, size: usize) -> Result<TensorData> {
        let absolute_offset = self.tensor_data_offset.checked_add(offset).ok_or_else(|| {
            GGUFError::InvalidTensorData("Tensor offset overflows u64".to_string())
        })?;
        let size_u64 = u64::try_from(size).map_err(|_| {
            GGUFError::InvalidTensorData("Tensor size does not fit u64".to_string())
        })?;
        let absolute_end = absolute_offset.checked_add(size_u64).ok_or_else(|| {
            GGUFError::InvalidTensorData("Tensor range overflows u64".to_string())
        })?;
        if absolute_end > self.file_size {
            return Err(GGUFError::UnexpectedEof);
        }
        self.reader.seek(SeekFrom::Start(absolute_offset))?;

        let data = match read_exact_owned(&mut self.reader, size, self.buffer_size) {
            Ok(data) => data,
            Err(error) => {
                if let Ok(position) = self.reader.stream_position() {
                    self.position = position;
                }
                return Err(error);
            }
        };
        self.position = absolute_end;

        Ok(TensorData::new_owned(data))
    }

    fn tensor_range_by_index(&self, tensor_index: usize) -> Result<(u64, usize)> {
        let tensor_info = self.tensor_infos.get(tensor_index).ok_or_else(|| {
            GGUFError::InvalidTensorData("Tensor descriptor index is out of bounds".to_string())
        })?;
        let data_size_u64 = tensor_info.checked_expected_data_size()?;
        let data_size = usize::try_from(data_size_u64).map_err(|_| {
            GGUFError::InvalidTensorData(format!(
                "Tensor '{}' size does not fit this platform",
                tensor_info.name()
            ))
        })?;
        let absolute_offset =
            self.tensor_data_offset.checked_add(tensor_info.data_offset()).ok_or_else(|| {
                GGUFError::InvalidTensorData("Tensor offset overflows u64".to_string())
            })?;
        let absolute_end = absolute_offset.checked_add(data_size_u64).ok_or_else(|| {
            GGUFError::InvalidTensorData("Tensor range overflows u64".to_string())
        })?;
        if absolute_end > self.file_size {
            return Err(GGUFError::UnexpectedEof);
        }
        Ok((absolute_offset, data_size))
    }

    fn read_exact_tracking(&mut self, mut buffer: &mut [u8]) -> Result<()> {
        while !buffer.is_empty() {
            match self.reader.read(buffer) {
                Ok(0) => return Err(GGUFError::UnexpectedEof),
                Ok(bytes_read) => {
                    self.position = self
                        .position
                        .checked_add(u64::try_from(bytes_read).map_err(|_| {
                            GGUFError::InvalidTensorData("Read size does not fit u64".to_string())
                        })?)
                        .ok_or_else(|| {
                            GGUFError::InvalidTensorData(
                                "File reader position overflows u64".to_string(),
                            )
                        })?;
                    let (_, remaining) = buffer.split_at_mut(bytes_read);
                    buffer = remaining;
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    /// Get current position in file
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Get tensor data section offset
    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    /// Get the alignment required for tensor offsets.
    pub fn tensor_alignment(&self) -> u64 {
        self.tensor_alignment
    }

    /// Validate structural counts, descriptors, alignment, and payload ranges.
    ///
    /// This does not read payload bytes or verify a cryptographic checksum. Use
    /// [`Self::validate_all_tensor_data`] when every declared range must also be
    /// read from the source.
    pub fn validate_integrity(&mut self) -> Result<()> {
        // Check header consistency
        if u64::try_from(self.tensor_infos.len()).ok() != Some(self.header.tensor_count) {
            return Err(GGUFError::Format(
                "Header tensor count doesn't match actual tensor count".to_string(),
            ));
        }

        if u64::try_from(self.metadata.len()).ok() != Some(self.header.metadata_kv_count) {
            return Err(GGUFError::Format(
                "Header metadata count doesn't match actual metadata count".to_string(),
            ));
        }

        // Validate tensor infos
        for tensor_info in &self.tensor_infos {
            tensor_info.validate()?;
        }

        // Check for tensor offset overlaps
        let mut tensor_ranges = Vec::new();
        tensor_ranges.try_reserve(self.tensor_infos.len()).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor range list".to_string())
        })?;
        for tensor in &self.tensor_infos {
            if !tensor.data_offset().is_multiple_of(self.tensor_alignment) {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' offset {} is not aligned to {} bytes",
                    tensor.name(),
                    tensor.data_offset(),
                    self.tensor_alignment
                )));
            }
            let size = tensor.checked_expected_data_size()?;
            let relative_end = tensor.data_offset().checked_add(size).ok_or_else(|| {
                GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' range overflows u64",
                    tensor.name()
                ))
            })?;
            let absolute_end =
                self.tensor_data_offset.checked_add(relative_end).ok_or_else(|| {
                    GGUFError::InvalidTensorData(format!(
                        "Tensor '{}' absolute range overflows u64",
                        tensor.name()
                    ))
                })?;
            if absolute_end > self.file_size {
                return Err(GGUFError::UnexpectedEof);
            }
            tensor_ranges.push((tensor.data_offset(), size, tensor.name()));
        }

        tensor_ranges.sort_unstable_by_key(|(offset, _, _)| *offset);

        for window in tensor_ranges.windows(2) {
            let (start_offset1, size1, name1) = window[0];
            let (start_offset2, _, name2) = window[1];

            let end_offset1 = start_offset1.checked_add(size1).ok_or_else(|| {
                GGUFError::InvalidTensorData(format!("Tensor '{}' range overflows u64", name1))
            })?;
            if end_offset1 > start_offset2 {
                return Err(GGUFError::Format(format!(
                    "Tensor data overlap detected: '{}' ({}..{}) overlaps with '{}' ({}..)",
                    name1, start_offset1, end_offset1, name2, start_offset2
                )));
            }
        }

        Ok(())
    }

    /// Get a summary of the GGUF file
    pub fn summary(&self) -> GGUFFileSummary {
        let total_tensor_size = self
            .tensor_infos
            .iter()
            .fold(0u64, |total, tensor| total.saturating_add(tensor.expected_data_size()));

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
        let mut total_loaded_bytes = 0usize;
        let mut total_expected_bytes = 0usize;
        let mut tensor_info_size = 0usize;

        for tensor_info in &self.tensor_infos {
            tensor_info_size =
                tensor_info_size.saturating_add(tensor_info.name().len()).saturating_add(32); // Approximate descriptor fields and allocation metadata.
            total_expected_bytes = total_expected_bytes.saturating_add(
                tensor_info
                    .checked_expected_data_size()
                    .ok()
                    .and_then(|size| usize::try_from(size).ok())
                    .unwrap_or(usize::MAX),
            );
            if let Some(data) = tensor_info.data() {
                total_loaded_bytes = total_loaded_bytes.saturating_add(data.len());
            }
        }

        // The lookup table deliberately owns one cloned key per tensor so name
        // lookups remain O(1). Include that cost in the reported reader overhead.
        for name in self.tensor_name_index.keys() {
            tensor_info_size = tensor_info_size
                .saturating_add(name.len())
                .saturating_add(std::mem::size_of::<String>())
                .saturating_add(std::mem::size_of::<usize>());
        }

        GGUFMemoryUsage {
            header_size: GGUFHeader::size(),
            metadata_size: self.metadata.serialized_size(),
            tensor_info_size,
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
        for tensor_index in 0..self.tensor_infos.len() {
            if predicate(self.tensor_infos[tensor_index].name()) {
                let data = self.load_tensor_data_by_index(tensor_index)?;
                self.tensor_infos[tensor_index].set_data(data);
                loaded_count += 1;
            }
        }

        Ok(loaded_count)
    }

    /// Get underlying reader (consuming the GGUFFileReader)
    pub fn into_inner(self) -> R {
        self.reader
    }
}

fn read_exact_owned<R: Read>(reader: &mut R, size: usize, buffer_size: usize) -> Result<Vec<u8>> {
    if size == 0 {
        return Ok(Vec::new());
    }
    if buffer_size == 0 {
        return Err(GGUFError::InvalidTensorData(
            "File reader buffer size must be greater than zero".to_string(),
        ));
    }

    let chunk_size = size.min(buffer_size).min(MAX_READER_CHUNK_SIZE);
    let mut chunk = Vec::new();
    chunk.try_reserve_exact(chunk_size).map_err(|_| {
        GGUFError::InvalidTensorData("Unable to allocate tensor read chunk".to_string())
    })?;
    chunk.resize(chunk_size, 0);
    let mut data = Vec::new();
    let mut remaining = size;
    while remaining > 0 {
        let to_read = remaining.min(chunk.len());
        reader.read_exact(&mut chunk[..to_read])?;
        data.try_reserve(to_read).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor data buffer".to_string())
        })?;
        data.extend_from_slice(&chunk[..to_read]);
        remaining -= to_read;
    }
    Ok(data)
}

fn try_zeroed_buffer(size: usize, description: &str) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(size)
        .map_err(|_| GGUFError::InvalidTensorData(format!("Unable to allocate {description}")))?;
    buffer.resize(size, 0);
    Ok(buffer)
}

fn try_reserve_vec_slot<T>(values: &mut Vec<T>, total: usize, description: &str) -> Result<()> {
    if values.len() < values.capacity() {
        return Ok(());
    }
    let target = values.capacity().saturating_mul(2).max(MIN_TENSOR_TRACKING_CAPACITY).min(total);
    values
        .try_reserve_exact(target.saturating_sub(values.len()))
        .map_err(|_| GGUFError::InvalidTensorData(format!("Unable to allocate {description}")))
}

fn try_reserve_map_slot<K: std::hash::Hash + Eq, V>(
    values: &mut HashMap<K, V>,
    total: usize,
    description: &str,
) -> Result<()> {
    if values.len() < values.capacity() {
        return Ok(());
    }
    let target = values.capacity().saturating_mul(2).max(MIN_TENSOR_TRACKING_CAPACITY).min(total);
    values
        .try_reserve(target.saturating_sub(values.len()))
        .map_err(|_| GGUFError::InvalidTensorData(format!("Unable to allocate {description}")))
}

fn try_clone_tensor_name(name: &str) -> Result<String> {
    let mut owned = String::new();
    owned
        .try_reserve_exact(name.len())
        .map_err(|_| GGUFError::InvalidTensorData("Unable to allocate tensor name".to_string()))?;
    owned.push_str(name);
    Ok(owned)
}

fn checked_align_u64(position: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 {
        return Err(GGUFError::InvalidMetadata("Tensor alignment cannot be zero".to_string()));
    }
    let remainder = position % alignment;
    let padding = if remainder == 0 { 0 } else { alignment - remainder };
    position
        .checked_add(padding)
        .ok_or_else(|| GGUFError::Format("Tensor data offset overflows u64".to_string()))
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
    /// Approximate size of tensor descriptors and the tensor-name lookup index
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

    /// Return the fraction of expected tensor bytes currently loaded in memory.
    pub fn loaded_fraction(&self) -> f32 {
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
        writeln!(f, "  Loaded fraction: {:.2}%", self.loaded_fraction() * 100.0)?;

        Ok(())
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::format::constants::*;
    use std::cell::Cell;
    use std::io::Cursor;
    use std::rc::Rc;

    #[derive(Debug)]
    struct PayloadReadTracker {
        cursor: Cursor<Vec<u8>>,
        payload_start: u64,
        payload_read: Rc<Cell<bool>>,
    }

    impl Read for PayloadReadTracker {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let start = self.cursor.position();
            let bytes_read = self.cursor.read(buffer)?;
            if start.saturating_add(bytes_read as u64) > self.payload_start {
                self.payload_read.set(true);
            }
            Ok(bytes_read)
        }
    }

    impl Seek for PayloadReadTracker {
        fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
            self.cursor.seek(position)
        }
    }

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

    fn create_duplicate_tensor_gguf_data() -> Vec<u8> {
        let mut data = create_minimal_gguf_data();
        data[8..16].copy_from_slice(&2u64.to_le_bytes());
        let tensor_name = data.windows(11).position(|window| window == b"test_tensor").unwrap();
        let descriptor_start = tensor_name - 8;
        let descriptor_end = tensor_name + 11 + 4 + (2 * 8) + 4 + 8;
        let duplicate = data[descriptor_start..descriptor_end].to_vec();
        data.splice(descriptor_end..descriptor_end, duplicate);
        data
    }

    fn create_many_tensor_gguf_data(count: usize) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&(count as u64).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        for index in 0..count {
            let name = format!("tensor_{index:04}");
            data.extend_from_slice(&(name.len() as u64).to_le_bytes());
            data.extend_from_slice(name.as_bytes());
            data.extend_from_slice(&1u32.to_le_bytes());
            data.extend_from_slice(&1u64.to_le_bytes());
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&((index as u64) * 32).to_le_bytes());
        }
        while data.len() % 32 != 0 {
            data.push(0);
        }
        for index in 0..count {
            data.extend_from_slice(&[index as u8; 4]);
            if index + 1 < count {
                data.extend_from_slice(&[0; 28]);
            }
        }
        data
    }

    fn create_overlapping_tensor_gguf_data() -> (Vec<u8>, u64) {
        let mut data = create_many_tensor_gguf_data(2);
        let second_name = b"tensor_0001";
        let name_position = data
            .windows(second_name.len())
            .position(|window| window == second_name)
            .unwrap();
        let offset_position = name_position + second_name.len() + 4 + 8 + 4;
        data[offset_position..offset_position + 8].copy_from_slice(&0u64.to_le_bytes());

        // The fixture stores 4 bytes for each tensor with a 28-byte alignment gap.
        let payload_start = (data.len() - 36) as u64;
        (data, payload_start)
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
    fn test_duplicate_tensor_names_are_rejected() {
        let error =
            GGUFFileReader::new(Cursor::new(create_duplicate_tensor_gguf_data())).unwrap_err();
        assert!(matches!(
            error,
            GGUFError::InvalidTensorData(message) if message.contains("Duplicate tensor name")
        ));
    }

    #[test]
    fn test_large_name_index_preserves_descriptor_order_and_lookup() {
        const TENSOR_COUNT: usize = 256;
        let mut reader =
            GGUFFileReader::new(Cursor::new(create_many_tensor_gguf_data(TENSOR_COUNT))).unwrap();
        assert_eq!(reader.tensor_name_index.len(), TENSOR_COUNT);

        for index in (0..TENSOR_COUNT).rev() {
            let name = format!("tensor_{index:04}");
            let descriptor = reader.get_tensor_info(&name).unwrap();
            assert_eq!(descriptor.name(), name);
            assert_eq!(descriptor.data_offset(), (index as u64) * 32);
            let payload = reader.load_tensor_data(&name).unwrap().unwrap();
            assert_eq!(payload.as_slice(), &[index as u8; 4]);
        }
        assert!(reader.get_tensor_info("missing").is_none());
    }

    #[test]
    fn test_gguf_reader_config() {
        let data = create_minimal_gguf_data();
        let cursor = Cursor::new(data);

        let config = GGUFReaderConfig {
            validate_integrity: true,
            eager_load_tensors: false,
            max_file_size: 1024,
            max_metadata_size: 1024,
            max_decoded_metadata_size: 1024,
            buffer_size: 8192,
            use_mmap: false,
        };

        let reader = GGUFFileReader::with_config(cursor, config).unwrap();
        assert!(!reader.is_fully_loaded());

        let unsupported = GGUFReaderConfig { use_mmap: true, ..Default::default() };
        assert!(matches!(
            GGUFFileReader::with_config(Cursor::new(create_minimal_gguf_data()), unsupported),
            Err(GGUFError::FeatureUnavailable(_))
        ));
    }

    #[test]
    fn test_file_metadata_budgets_are_configurable() {
        let serialized_error = GGUFFileReader::with_config(
            Cursor::new(create_minimal_gguf_data()),
            GGUFReaderConfig { max_metadata_size: 1, ..Default::default() },
        )
        .unwrap_err();
        assert!(serialized_error.to_string().contains("Metadata exceeds byte limit"));

        let decoded_error = GGUFFileReader::with_config(
            Cursor::new(create_minimal_gguf_data()),
            GGUFReaderConfig { max_decoded_metadata_size: 0, ..Default::default() },
        )
        .unwrap_err();
        assert!(decoded_error.to_string().contains("Decoded metadata allocation exceeds budget"));
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
        assert_eq!(reader.position(), reader.tensor_data_offset() + 24);

        reader.read_tensor_data_at(0, 4).unwrap();
        assert_eq!(reader.position(), reader.tensor_data_offset() + 4);
    }

    #[test]
    fn test_payload_validation_streams_without_retaining_data() {
        let config = GGUFReaderConfig { buffer_size: 3, ..Default::default() };
        let mut reader =
            GGUFFileReader::with_config(Cursor::new(create_minimal_gguf_data()), config).unwrap();

        reader.validate_all_tensor_data().unwrap();
        assert!(!reader.is_fully_loaded());
        assert!(reader.tensor_infos().iter().all(|tensor| !tensor.has_data()));
        assert_eq!(reader.position(), reader.tensor_data_offset() + 24);
    }

    #[test]
    fn test_payload_comparison_is_chunked_and_does_not_retain_data() {
        let config = GGUFReaderConfig { buffer_size: 3, ..Default::default() };
        let mut left =
            GGUFFileReader::with_config(Cursor::new(create_minimal_gguf_data()), config.clone())
                .unwrap();
        let mut equal =
            GGUFFileReader::with_config(Cursor::new(create_minimal_gguf_data()), config.clone())
                .unwrap();
        assert!(left.tensor_data_equals("test_tensor", &mut equal).unwrap());

        let mut different_data = create_minimal_gguf_data();
        *different_data.last_mut().unwrap() = 1;
        let mut different =
            GGUFFileReader::with_config(Cursor::new(different_data), config).unwrap();
        assert!(!left.tensor_data_equals("test_tensor", &mut different).unwrap());
        assert!(left.tensor_infos().iter().all(|tensor| !tensor.has_data()));
        assert!(different.tensor_infos().iter().all(|tensor| !tensor.has_data()));
    }

    #[test]
    fn test_tensor_tracking_growth_is_bounded() {
        let mut descriptors = Vec::<u8>::new();
        try_reserve_vec_slot(&mut descriptors, usize::MAX, "test descriptors").unwrap();
        assert!(descriptors.capacity() >= MIN_TENSOR_TRACKING_CAPACITY);
        assert!(descriptors.capacity() < 1024);

        let mut names = HashMap::<u8, usize>::new();
        try_reserve_map_slot(&mut names, usize::MAX, "test names").unwrap();
        assert!(names.capacity() >= MIN_TENSOR_TRACKING_CAPACITY);
        assert!(names.capacity() < 1024);
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
        assert_eq!(memory_usage.loaded_fraction(), 0.0);
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
    fn test_integrity_validation_precedes_eager_payload_reads() {
        let (data, payload_start) = create_overlapping_tensor_gguf_data();
        let payload_read = Rc::new(Cell::new(false));
        let reader = PayloadReadTracker {
            cursor: Cursor::new(data),
            payload_start,
            payload_read: Rc::clone(&payload_read),
        };
        let config = GGUFReaderConfig {
            eager_load_tensors: true,
            validate_integrity: true,
            ..Default::default()
        };

        let error = GGUFFileReader::with_config(reader, config).unwrap_err();
        assert!(error.to_string().contains("overlap"));
        assert!(!payload_read.get(), "malformed descriptors must fail before payload reads");
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
