//! Stream-based GGUF reader for non-seekable streams

use crate::error::{GGUFError, Result};
use crate::format::constants::GGUF_MAX_METADATA_DECODED_SIZE;
use crate::format::types::GGUFTensorType as TensorType;
use crate::format::{GGUFHeader, Metadata, TensorInfo};
use crate::tensor::{TensorData, TensorInfo as TensorInfoNew, TensorShape};
use std::collections::HashMap;
use std::io::{BufReader, Read};

const MIN_TENSOR_TRACKING_CAPACITY: usize = 64;
const MAX_READER_CHUNK_SIZE: usize = 1024 * 1024;

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
    /// Tensor descriptor index by name
    tensor_name_index: HashMap<String, usize>,
    /// Current position in the stream
    position: u64,
    /// Whether we've reached the tensor data section
    at_tensor_data: bool,
    /// Absolute start of the tensor data section
    tensor_data_offset: u64,
    /// Relative byte span from the data-section start through the last tensor
    tensor_data_span: u64,
    /// Required alignment for tensor offsets
    tensor_alignment: u64,
    /// Maximum temporary buffer used while growing owned tensor data
    tensor_buffer_size: usize,
    /// Descriptor indices ordered by tensor data offset, then descriptor order
    tensor_order: Vec<usize>,
    /// Cursor into `tensor_order` for the next tensor to read
    next_tensor: usize,
}

/// Configuration for stream reading
#[derive(Debug, Clone)]
pub struct StreamReaderConfig {
    /// Preferred buffer size for tensor reads; temporary chunks are capped at 1 MiB.
    pub buffer_size: usize,
    /// Request checksum validation.
    ///
    /// GGUF has no standard checksum field; setting this to `true` returns
    /// [`GGUFError::FeatureUnavailable`].
    pub validate_checksums: bool,
    /// Maximum number of serialized metadata bytes to read.
    pub max_metadata_size: usize,
    /// Maximum decoded allocation budget for metadata.
    pub max_decoded_metadata_size: usize,
    /// Maximum number of tensors to prevent DoS
    pub max_tensor_count: usize,
}

impl Default for StreamReaderConfig {
    fn default() -> Self {
        Self {
            buffer_size: 64 * 1024,              // 64KB
            validate_checksums: false,           // GGUF v3 has no checksum field
            max_metadata_size: 16 * 1024 * 1024, // 16MB
            max_decoded_metadata_size: GGUF_MAX_METADATA_DECODED_SIZE,
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
        if config.validate_checksums {
            return Err(GGUFError::FeatureUnavailable("stream checksum validation".to_string()));
        }
        let mut position = 0u64;

        // Read header
        let header = GGUFHeader::read_from(&mut reader)?;
        header.validate_comprehensive()?;
        position = position
            .checked_add(GGUFHeader::size() as u64)
            .ok_or_else(|| GGUFError::Format("Stream position overflow".to_string()))?;

        // Check limits
        let max_tensor_count = u64::try_from(config.max_tensor_count).unwrap_or(u64::MAX);
        if header.tensor_count > max_tensor_count {
            return Err(GGUFError::Format(format!(
                "Too many tensors: {} exceeds limit of {}",
                header.tensor_count, config.max_tensor_count
            )));
        }

        // Read metadata
        let metadata = Metadata::read_from_with_limits(
            &mut reader,
            header.metadata_kv_count,
            config.max_metadata_size,
            config.max_decoded_metadata_size,
        )?;
        let metadata_size = metadata.checked_serialized_size().ok_or_else(|| {
            GGUFError::InvalidMetadata("Serialized metadata size overflows usize".to_string())
        })?;
        position = position
            .checked_add(u64::try_from(metadata_size).map_err(|_| {
                GGUFError::InvalidMetadata("Metadata size does not fit u64".to_string())
            })?)
            .ok_or_else(|| GGUFError::Format("Stream position overflow".to_string()))?;
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
            let descriptor_size = tensor_info.checked_serialized_size().ok_or_else(|| {
                GGUFError::InvalidTensorData("Tensor descriptor size overflows usize".to_string())
            })?;
            position = position
                .checked_add(u64::try_from(descriptor_size).map_err(|_| {
                    GGUFError::InvalidTensorData(
                        "Tensor descriptor size does not fit u64".to_string(),
                    )
                })?)
                .ok_or_else(|| GGUFError::Format("Stream position overflow".to_string()))?;

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

        let mut tensor_order = Vec::new();
        tensor_order.try_reserve_exact(tensor_capacity).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor order".to_string())
        })?;
        tensor_order.extend(0..tensor_infos.len());
        tensor_order.sort_unstable_by_key(|&index| (tensor_infos[index].data_offset(), index));

        // Calculate alignment padding
        let tensor_data_offset = checked_align_u64(position, tensor_alignment)?;
        let tensor_data_span =
            validate_tensor_layout(&tensor_infos, &tensor_order, tensor_data_offset)?;
        let padding_size = usize::try_from(tensor_data_offset - position)
            .map_err(|_| GGUFError::Format("Alignment padding does not fit usize".to_string()))?;
        if padding_size > 0 {
            read_discard(&mut reader, padding_size)?;
            position = tensor_data_offset;
        }

        Ok(Self {
            reader,
            header,
            metadata,
            tensor_infos,
            tensor_name_index,
            position,
            at_tensor_data: true,
            tensor_data_offset,
            tensor_data_span,
            tensor_alignment,
            tensor_buffer_size: config.buffer_size,
            tensor_order,
            next_tensor: 0,
        })
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

    /// Get tensor names in order
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_infos.iter().map(|t| t.name()).collect()
    }

    /// Get the number of tensors
    pub fn tensor_count(&self) -> usize {
        self.tensor_infos.len()
    }

    /// Return the tensor alignment declared by file metadata.
    pub fn tensor_alignment(&self) -> u64 {
        self.tensor_alignment
    }

    /// Read the next tensor's data in stream order.
    ///
    /// Tensor payloads are returned as owned data and are not retained in the
    /// descriptor list, keeping reader memory bounded by the current tensor.
    pub fn read_next_tensor(&mut self) -> Result<Option<(String, TensorData)>> {
        if !self.at_tensor_data {
            return Ok(None);
        }

        if let Some(&index) = self.tensor_order.get(self.next_tensor) {
            let tensor_info = &self.tensor_infos[index];
            let data_size_u64 = tensor_info.checked_expected_data_size()?;
            let data_size = usize::try_from(data_size_u64).map_err(|_| {
                GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' size does not fit this platform",
                    tensor_info.name()
                ))
            })?;
            let target_position =
                self.tensor_data_offset.checked_add(tensor_info.data_offset()).ok_or_else(
                    || GGUFError::InvalidTensorData("Tensor offset overflows u64".to_string()),
                )?;
            let tensor_name = try_clone_tensor_name(tensor_info.name())?;

            // Skip to the right position if needed
            if self.position < target_position {
                let skip_bytes =
                    usize::try_from(target_position - self.position).map_err(|_| {
                        GGUFError::InvalidTensorData("Tensor gap does not fit usize".to_string())
                    })?;
                self.skip_bytes(skip_bytes)?;
            } else if self.position > target_position {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' starts at {}, before current stream position {}",
                    tensor_name, target_position, self.position
                )));
            }

            // Read tensor data
            let data = read_exact_owned(&mut self.reader, data_size, self.tensor_buffer_size)?;
            self.position = self.position.checked_add(data_size_u64).ok_or_else(|| {
                GGUFError::InvalidTensorData("Stream position overflows u64".to_string())
            })?;

            self.next_tensor = self.next_tensor.checked_add(1).ok_or_else(|| {
                GGUFError::InvalidTensorData("Tensor stream cursor overflows usize".to_string())
            })?;

            Ok(Some((tensor_name, TensorData::new_owned(data))))
        } else {
            Ok(None)
        }
    }

    /// Read all tensors in stream order
    pub fn read_all_tensors(&mut self) -> Result<HashMap<String, TensorData>> {
        let mut tensors = HashMap::new();
        let remaining = self.tensor_order.len().saturating_sub(self.next_tensor);
        tensors.try_reserve(remaining).map_err(|_| {
            GGUFError::InvalidTensorData("Unable to allocate tensor data map".to_string())
        })?;

        while let Some((name, data)) = self.read_next_tensor()? {
            tensors.insert(name, data);
        }

        Ok(tensors)
    }

    /// Skip a certain number of bytes in the stream
    fn skip_bytes(&mut self, count: usize) -> Result<()> {
        const SKIP_BUFFER_SIZE: usize = 8192;
        let mut buffer = [0u8; SKIP_BUFFER_SIZE];
        let mut remaining = count;

        while remaining > 0 {
            let to_read = remaining.min(SKIP_BUFFER_SIZE);
            self.reader.read_exact(&mut buffer[..to_read])?;
            remaining -= to_read;
            self.position = self
                .position
                .checked_add(to_read as u64)
                .ok_or_else(|| GGUFError::Format("Stream position overflow".to_string()))?;
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

        let total_tensor_size = self
            .tensor_infos
            .iter()
            .fold(0u64, |total, tensor| total.saturating_add(tensor.expected_data_size()));

        StreamReaderSummary {
            header: self.header.clone(),
            metadata_count: self.metadata.len(),
            tensor_count: self.tensor_infos.len(),
            total_tensor_size,
            current_position: self.position,
            tensor_data_offset: self.tensor_data_offset,
            tensor_data_span: self.tensor_data_span,
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

fn read_discard<R: Read>(reader: &mut R, mut size: usize) -> Result<()> {
    const BUFFER_SIZE: usize = 8192;
    let mut buffer = [0u8; BUFFER_SIZE];
    while size > 0 {
        let chunk = size.min(BUFFER_SIZE);
        reader.read_exact(&mut buffer[..chunk])?;
        size -= chunk;
    }
    Ok(())
}

fn read_exact_owned<R: Read>(reader: &mut R, size: usize, buffer_size: usize) -> Result<Vec<u8>> {
    if size == 0 {
        return Ok(Vec::new());
    }
    if buffer_size == 0 {
        return Err(GGUFError::InvalidTensorData(
            "Stream reader buffer size must be greater than zero".to_string(),
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

fn try_clone_tensor_name(name: &str) -> Result<String> {
    let mut owned = String::new();
    owned
        .try_reserve_exact(name.len())
        .map_err(|_| GGUFError::InvalidTensorData("Unable to allocate tensor name".to_string()))?;
    owned.push_str(name);
    Ok(owned)
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

fn validate_tensor_layout(
    tensor_infos: &[TensorInfoNew],
    tensor_order: &[usize],
    tensor_data_offset: u64,
) -> Result<u64> {
    let mut previous: Option<(u64, &str)> = None;
    let mut tensor_data_span = 0u64;
    for &index in tensor_order {
        let tensor = tensor_infos.get(index).ok_or_else(|| {
            GGUFError::InvalidTensorData("Tensor order index is out of bounds".to_string())
        })?;
        let end = tensor
            .data_offset()
            .checked_add(tensor.checked_expected_data_size()?)
            .ok_or_else(|| {
                GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' range overflows u64",
                    tensor.name()
                ))
            })?;
        tensor_data_offset.checked_add(end).ok_or_else(|| {
            GGUFError::InvalidTensorData(format!(
                "Tensor '{}' absolute range overflows u64",
                tensor.name()
            ))
        })?;
        if let Some((previous_end, previous_name)) = previous {
            if previous_end > tensor.data_offset() {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Tensor data overlap detected: '{}' ends at {}, after '{}' starts at {}",
                    previous_name,
                    previous_end,
                    tensor.name(),
                    tensor.data_offset()
                )));
            }
        }
        tensor_data_span = tensor_data_span.max(end);
        previous = Some((end, tensor.name()));
    }
    Ok(tensor_data_span)
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
    /// Absolute start of the tensor data section
    pub tensor_data_offset: u64,
    /// Relative span through the end of the last tensor, including gaps
    pub tensor_data_span: u64,
    /// Count of each tensor type
    pub tensor_types: HashMap<TensorType, usize>,
}

impl StreamReaderSummary {
    /// Percentage of the tensor-data span consumed by the stream reader.
    pub fn progress_percentage(&self) -> f64 {
        if self.tensor_data_span == 0 {
            return 100.0;
        }

        let consumed = self
            .current_position
            .saturating_sub(self.tensor_data_offset)
            .min(self.tensor_data_span);
        (consumed as f64 / self.tensor_data_span as f64) * 100.0
    }
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
        writeln!(f, "  Tensor data span: {} bytes", self.tensor_data_span)?;
        writeln!(f, "  Current position: {} bytes", self.current_position)?;
        writeln!(f, "  Progress: {:.1}%", self.progress_percentage())?;
        writeln!(f, "  Tensor types:")?;

        for (tensor_type, count) in &self.tensor_types {
            writeln!(f, "    {}: {}", tensor_type.name(), count)?;
        }

        Ok(())
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
        .ok_or_else(|| GGUFError::Format("Tensor data offset overflows u64".to_string()))
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

        // Tensor offsets are relative to the aligned tensor data section.
        let tensor_a_pos = offset_positions[0];
        data[tensor_a_pos..tensor_a_pos + 8].copy_from_slice(&0u64.to_le_bytes());

        // Tensor B starts on the next 32-byte boundary after tensor A.
        let tensor_b_pos = offset_positions[1];
        data[tensor_b_pos..tensor_b_pos + 8].copy_from_slice(&32u64.to_le_bytes());

        // Tensor data A (4 F32 = 16 bytes)
        data.extend_from_slice(&[0xAA; 16]);

        // Inter-tensor alignment padding
        data.extend_from_slice(&[0u8; 16]);

        // Tensor data B (3 F32 = 12 bytes)
        data.extend_from_slice(&[0xBB; 12]);

        data
    }

    fn create_out_of_order_stream_gguf_data() -> Vec<u8> {
        let mut data = create_stream_gguf_data();
        let tensor_a_name = data.windows(8).position(|window| window == b"tensor_a").unwrap();
        let tensor_b_name = data.windows(8).position(|window| window == b"tensor_b").unwrap();
        let tensor_a_start = tensor_a_name - 8;
        let tensor_b_start = tensor_b_name - 8;
        let descriptor_size = tensor_b_start - tensor_a_start;
        let tensor_a = data[tensor_a_start..tensor_a_start + descriptor_size].to_vec();
        let tensor_b = data[tensor_b_start..tensor_b_start + descriptor_size].to_vec();
        data[tensor_a_start..tensor_a_start + descriptor_size].copy_from_slice(&tensor_b);
        data[tensor_b_start..tensor_b_start + descriptor_size].copy_from_slice(&tensor_a);
        data
    }

    fn set_tensor_offset(data: &mut [u8], name: &[u8; 8], offset: u64) {
        let name_position = data.windows(name.len()).position(|window| window == name).unwrap();
        let offset_position = name_position + name.len() + 4 + 8 + 4;
        data[offset_position..offset_position + 8].copy_from_slice(&offset.to_le_bytes());
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
            validate_checksums: false,
            max_metadata_size: 1024,
            max_decoded_metadata_size: 1024,
            max_tensor_count: 10,
        };

        let reader = GGUFStreamReader::with_config(cursor, config).unwrap();
        assert_eq!(reader.tensor_count(), 2);

        let unsupported = StreamReaderConfig { validate_checksums: true, ..Default::default() };
        assert!(matches!(
            GGUFStreamReader::with_config(Cursor::new(create_stream_gguf_data()), unsupported),
            Err(GGUFError::FeatureUnavailable(_))
        ));
    }

    #[test]
    fn test_stream_metadata_budgets_are_configurable() {
        let serialized_error = GGUFStreamReader::with_config(
            Cursor::new(create_stream_gguf_data()),
            StreamReaderConfig { max_metadata_size: 1, ..Default::default() },
        )
        .unwrap_err();
        assert!(serialized_error.to_string().contains("Metadata exceeds byte limit"));

        let decoded_error = GGUFStreamReader::with_config(
            Cursor::new(create_stream_gguf_data()),
            StreamReaderConfig { max_decoded_metadata_size: 0, ..Default::default() },
        )
        .unwrap_err();
        assert!(decoded_error.to_string().contains("Decoded metadata allocation exceeds budget"));
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
    fn test_read_next_tensor_uses_precomputed_data_order() {
        let data = create_out_of_order_stream_gguf_data();
        let mut reader = GGUFStreamReader::new(Cursor::new(data)).unwrap();

        assert_eq!(reader.tensor_order, [1, 0]);
        assert_eq!(reader.tensor_name_index["tensor_b"], 0);
        assert_eq!(reader.tensor_name_index["tensor_a"], 1);
        let (first_name, first_data) = reader.read_next_tensor().unwrap().unwrap();
        assert_eq!(first_name, "tensor_a");
        assert_eq!(first_data.as_slice(), &[0xAA; 16]);
        assert_eq!(reader.next_tensor, 1);

        let (second_name, second_data) = reader.read_next_tensor().unwrap().unwrap();
        assert_eq!(second_name, "tensor_b");
        assert_eq!(second_data.as_slice(), &[0xBB; 12]);
        assert_eq!(reader.next_tensor, 2);
        assert!(reader.read_next_tensor().unwrap().is_none());
    }

    #[test]
    fn test_streamed_payload_is_not_retained_by_reader() {
        let data = create_stream_gguf_data();
        let mut reader = GGUFStreamReader::new(Cursor::new(data)).unwrap();

        let (name, returned) = reader.read_next_tensor().unwrap().unwrap();
        assert!(matches!(returned, TensorData::Owned(_)));
        assert!(reader.get_tensor_info(&name).unwrap().data().is_none());
        assert!(reader.tensor_infos().iter().all(|tensor| !tensor.has_data()));
    }

    #[test]
    fn test_duplicate_tensor_names_are_rejected() {
        let mut data = create_stream_gguf_data();
        let tensor_b_name = data.windows(8).position(|window| window == b"tensor_b").unwrap();
        data[tensor_b_name..tensor_b_name + 8].copy_from_slice(b"tensor_a");

        let error = GGUFStreamReader::new(Cursor::new(data)).unwrap_err();
        assert!(matches!(
            error,
            GGUFError::InvalidTensorData(message) if message.contains("Duplicate tensor name")
        ));
    }

    #[test]
    fn test_overlapping_tensor_ranges_are_rejected_during_construction() {
        let mut data = create_stream_gguf_data();
        set_tensor_offset(&mut data, b"tensor_b", 0);

        let error = GGUFStreamReader::new(Cursor::new(data)).unwrap_err();
        assert!(matches!(
            error,
            GGUFError::InvalidTensorData(message) if message.contains("overlap")
        ));
    }

    #[test]
    fn test_overflowing_tensor_range_is_rejected_during_construction() {
        let mut data = create_stream_gguf_data();
        set_tensor_offset(&mut data, b"tensor_a", u64::MAX - 31);

        let error = GGUFStreamReader::new(Cursor::new(data)).unwrap_err();
        assert!(matches!(
            error,
            GGUFError::InvalidTensorData(message) if message.contains("range overflows")
        ));
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

        let mut reader = GGUFStreamReader::new(cursor).unwrap();
        let summary = reader.summary();

        assert_eq!(summary.tensor_count, 2);
        assert_eq!(summary.metadata_count, 1);
        assert_eq!(summary.total_tensor_size, 28); // 16 + 12 bytes
        assert_eq!(summary.tensor_data_span, 44); // 16 bytes, 16-byte gap, 12 bytes
        assert_eq!(summary.current_position, summary.tensor_data_offset);
        assert_eq!(summary.progress_percentage(), 0.0);
        assert!(summary.tensor_types.contains_key(&TensorType::F32));

        while reader.read_next_tensor().unwrap().is_some() {}
        let completed = reader.summary();
        assert_eq!(completed.current_position, completed.tensor_data_offset + 44);
        assert_eq!(completed.progress_percentage(), 100.0);
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
