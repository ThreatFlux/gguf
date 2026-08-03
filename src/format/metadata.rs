//! GGUF metadata structures and operations

use crate::error::{GGUFError, Result};
use crate::format::constants::*;
use crate::format::types::GGUFValueType;

#[cfg(feature = "std")]
use crate::format::endian::{
    read_f32, read_f64, read_i16, read_i32, read_i64, read_i8, read_u16, read_u32, read_u64,
    read_u8, write_f32, write_f64, write_i16, write_i32, write_i64, write_i8, write_u16, write_u32,
    write_u64, write_u8,
};
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use std::io::{Read, Write};

#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec::Vec,
};
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use hashbrown::HashMap;

/// Variable-sized metadata is grown in bounded increments rather than trusting
/// attacker-controlled `u64` length fields as allocation sizes.
#[cfg(feature = "std")]
const METADATA_BYTE_READ_CHUNK: usize = 8 * 1024;
#[cfg(feature = "std")]
const METADATA_ARRAY_RESERVE_CHUNK: usize = 1024;

#[cfg(feature = "std")]
#[derive(Debug)]
struct MetadataDecodeBudget {
    remaining: usize,
    limit: usize,
}

#[cfg(feature = "std")]
impl MetadataDecodeBudget {
    fn new(limit: usize) -> Self {
        Self { remaining: limit, limit }
    }

    fn charge(&mut self, bytes: usize, description: &str) -> Result<()> {
        self.remaining = self.remaining.checked_sub(bytes).ok_or_else(|| {
            GGUFError::InvalidMetadata(format!(
                "Decoded metadata allocation exceeds budget of {} bytes while reading {}",
                self.limit, description
            ))
        })?;
        Ok(())
    }
}

// Import core modules for no_std compatibility
#[cfg(not(feature = "std"))]
use core::{fmt, slice};

/// A metadata value in a GGUF file
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// 8-bit unsigned integer
    U8(u8),
    /// 8-bit signed integer
    I8(i8),
    /// 16-bit unsigned integer
    U16(u16),
    /// 16-bit signed integer
    I16(i16),
    /// 32-bit unsigned integer
    U32(u32),
    /// 32-bit signed integer
    I32(i32),
    /// 32-bit floating point
    F32(f32),
    /// Boolean value
    Bool(bool),
    /// UTF-8 string
    String(String),
    /// Array of values
    Array(Box<MetadataArray>),
    /// 64-bit unsigned integer
    U64(u64),
    /// 64-bit signed integer
    I64(i64),
    /// 64-bit floating point
    F64(f64),
}

/// An array of metadata values (all of the same type)
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataArray {
    /// Type of elements in the array
    pub element_type: GGUFValueType,
    /// Number of elements
    pub length: u64,
    /// Array elements
    pub values: Vec<MetadataValue>,
}

/// Collection of metadata key-value pairs
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Metadata {
    /// Metadata key-value pairs
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub data: HashMap<String, MetadataValue>,
    #[cfg(not(any(feature = "std", feature = "alloc")))]
    pub data: (), // Placeholder for no_std + no_alloc builds
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl Default for Metadata {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(any(feature = "std", feature = "alloc")))]
impl Default for Metadata {
    fn default() -> Self {
        Self { data: () }
    }
}

impl MetadataValue {
    /// Get the type of this metadata value
    pub fn value_type(&self) -> GGUFValueType {
        match self {
            MetadataValue::U8(_) => GGUFValueType::U8,
            MetadataValue::I8(_) => GGUFValueType::I8,
            MetadataValue::U16(_) => GGUFValueType::U16,
            MetadataValue::I16(_) => GGUFValueType::I16,
            MetadataValue::U32(_) => GGUFValueType::U32,
            MetadataValue::I32(_) => GGUFValueType::I32,
            MetadataValue::F32(_) => GGUFValueType::F32,
            MetadataValue::Bool(_) => GGUFValueType::Bool,
            MetadataValue::String(_) => GGUFValueType::String,
            MetadataValue::Array(_) => GGUFValueType::Array,
            MetadataValue::U64(_) => GGUFValueType::U64,
            MetadataValue::I64(_) => GGUFValueType::I64,
            MetadataValue::F64(_) => GGUFValueType::F64,
        }
    }

    /// Calculate the serialized size of this value
    pub fn serialized_size(&self) -> usize {
        self.checked_serialized_size().unwrap_or(usize::MAX)
    }

    /// Calculate the serialized size without overflowing.
    pub fn checked_serialized_size(&self) -> Option<usize> {
        match self {
            MetadataValue::U8(_) | MetadataValue::I8(_) | MetadataValue::Bool(_) => Some(1),
            MetadataValue::U16(_) | MetadataValue::I16(_) => Some(2),
            MetadataValue::U32(_) | MetadataValue::I32(_) | MetadataValue::F32(_) => Some(4),
            MetadataValue::String(s) => 8usize.checked_add(s.len()),
            MetadataValue::Array(arr) => arr.as_ref().checked_serialized_size(),
            MetadataValue::U64(_) | MetadataValue::I64(_) | MetadataValue::F64(_) => Some(8),
        }
    }

    /// Validate length, type, and nesting invariants before serialization.
    pub fn validate(&self) -> Result<()> {
        self.validate_at_depth(0)
    }

    fn validate_at_depth(&self, depth: usize) -> Result<()> {
        match self {
            MetadataValue::Array(array) => array.validate_at_depth(depth + 1),
            _ => Ok(()),
        }
    }

    /// Read a metadata value from a reader
    #[cfg(feature = "std")]
    pub fn read_from<R: Read>(reader: &mut R, value_type: GGUFValueType) -> Result<Self> {
        let mut budget = MetadataDecodeBudget::new(GGUF_MAX_METADATA_DECODED_SIZE);
        Self::read_from_at_depth(reader, value_type, 0, &mut budget)
    }

    #[cfg(feature = "std")]
    fn read_from_at_depth<R: Read>(
        reader: &mut R,
        value_type: GGUFValueType,
        depth: usize,
        budget: &mut MetadataDecodeBudget,
    ) -> Result<Self> {
        let value = match value_type {
            GGUFValueType::U8 => MetadataValue::U8(read_u8(reader)?),
            GGUFValueType::I8 => MetadataValue::I8(read_i8(reader)?),
            GGUFValueType::U16 => MetadataValue::U16(read_u16(reader)?),
            GGUFValueType::I16 => MetadataValue::I16(read_i16(reader)?),
            GGUFValueType::U32 => MetadataValue::U32(read_u32(reader)?),
            GGUFValueType::I32 => MetadataValue::I32(read_i32(reader)?),
            GGUFValueType::F32 => MetadataValue::F32(read_f32(reader)?),
            GGUFValueType::Bool => match read_u8(reader)? {
                0 => MetadataValue::Bool(false),
                1 => MetadataValue::Bool(true),
                value => {
                    return Err(GGUFError::InvalidMetadata(format!(
                        "Invalid boolean value: {} (expected 0 or 1)",
                        value
                    )));
                }
            },
            GGUFValueType::String => {
                let len = read_u64(reader)?;
                let bytes = read_variable_bytes(reader, len, "string", budget)?;
                let string = String::from_utf8(bytes)
                    .map_err(|e| GGUFError::Format(format!("Invalid UTF-8 string: {}", e)))?;
                MetadataValue::String(string)
            }
            GGUFValueType::Array => {
                if depth >= GGUF_MAX_METADATA_NESTING_DEPTH {
                    return Err(GGUFError::InvalidMetadata(format!(
                        "Metadata array nesting exceeds limit of {}",
                        GGUF_MAX_METADATA_NESTING_DEPTH
                    )));
                }
                budget.charge(core::mem::size_of::<MetadataArray>(), "nested array")?;
                let array = MetadataArray::read_from_at_depth(reader, depth + 1, budget)?;
                MetadataValue::Array(Box::new(array))
            }
            GGUFValueType::U64 => MetadataValue::U64(read_u64(reader)?),
            GGUFValueType::I64 => MetadataValue::I64(read_i64(reader)?),
            GGUFValueType::F64 => MetadataValue::F64(read_f64(reader)?),
        };

        Ok(value)
    }

    /// Write a metadata value to a writer  
    #[cfg(feature = "std")]
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.validate()?;
        match self {
            MetadataValue::U8(v) => write_u8(writer, *v)?,
            MetadataValue::I8(v) => write_i8(writer, *v)?,
            MetadataValue::U16(v) => write_u16(writer, *v)?,
            MetadataValue::I16(v) => write_i16(writer, *v)?,
            MetadataValue::U32(v) => write_u32(writer, *v)?,
            MetadataValue::I32(v) => write_i32(writer, *v)?,
            MetadataValue::F32(v) => write_f32(writer, *v)?,
            MetadataValue::Bool(v) => write_u8(writer, if *v { 1 } else { 0 })?,
            MetadataValue::String(s) => {
                let length = u64::try_from(s.len()).map_err(|_| {
                    GGUFError::InvalidMetadata(
                        "String length does not fit the GGUF u64 length field".to_string(),
                    )
                })?;
                write_u64(writer, length)?;
                writer.write_all(s.as_bytes())?;
            }
            MetadataValue::Array(arr) => arr.as_ref().write_to(writer)?,
            MetadataValue::U64(v) => write_u64(writer, *v)?,
            MetadataValue::I64(v) => write_i64(writer, *v)?,
            MetadataValue::F64(v) => write_f64(writer, *v)?,
        }
        Ok(())
    }

    /// Convert to a string representation for display
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn to_string_representation(&self) -> String {
        match self {
            MetadataValue::U8(v) => v.to_string(),
            MetadataValue::I8(v) => v.to_string(),
            MetadataValue::U16(v) => v.to_string(),
            MetadataValue::I16(v) => v.to_string(),
            MetadataValue::U32(v) => v.to_string(),
            MetadataValue::I32(v) => v.to_string(),
            MetadataValue::F32(v) => v.to_string(),
            MetadataValue::Bool(v) => v.to_string(),
            MetadataValue::String(s) => s.clone(),
            MetadataValue::Array(arr) => format!("Array[{}; {}]", arr.element_type, arr.length),
            MetadataValue::U64(v) => v.to_string(),
            MetadataValue::I64(v) => v.to_string(),
            MetadataValue::F64(v) => v.to_string(),
        }
    }

    /// Try to convert to a specific type
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetadataValue::U8(v) => Some(*v as u64),
            MetadataValue::U16(v) => Some(*v as u64),
            MetadataValue::U32(v) => Some(*v as u64),
            MetadataValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            MetadataValue::I8(v) => Some(*v as i64),
            MetadataValue::I16(v) => Some(*v as i64),
            MetadataValue::I32(v) => Some(*v as i64),
            MetadataValue::I64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetadataValue::F32(v) => Some(*v as f64),
            MetadataValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

impl MetadataArray {
    /// Create a new metadata array
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn new(element_type: GGUFValueType, values: Vec<MetadataValue>) -> Result<Self> {
        // Validate that all values have the same type as specified
        for value in &values {
            if value.value_type() != element_type {
                return Err(GGUFError::InvalidMetadata(format!(
                    "Array element type mismatch: expected {}, got {}",
                    element_type,
                    value.value_type()
                )));
            }
        }

        let length = u64::try_from(values.len()).map_err(|_| {
            GGUFError::InvalidMetadata(
                "Array length does not fit the GGUF u64 length field".to_string(),
            )
        })?;

        let array = Self { element_type, length, values };
        array.validate()?;
        Ok(array)
    }

    #[cfg(not(any(feature = "std", feature = "alloc")))]
    pub fn new(_element_type: GGUFValueType, _values: &[MetadataValue]) -> Result<Self> {
        Err(GGUFError::AllocationRequired)
    }

    /// Calculate the serialized size of this array
    pub fn serialized_size(&self) -> usize {
        self.checked_serialized_size().unwrap_or(usize::MAX)
    }

    /// Calculate the serialized array size without overflowing.
    pub fn checked_serialized_size(&self) -> Option<usize> {
        self.values
            .iter()
            .try_fold(12usize, |size, value| size.checked_add(value.checked_serialized_size()?))
    }

    /// Validate the array's length, homogeneous element type, and nesting depth.
    pub fn validate(&self) -> Result<()> {
        self.validate_at_depth(1)
    }

    fn validate_at_depth(&self, depth: usize) -> Result<()> {
        if depth > GGUF_MAX_METADATA_NESTING_DEPTH {
            return Err(GGUFError::InvalidMetadata(format!(
                "Metadata array nesting exceeds limit of {}",
                GGUF_MAX_METADATA_NESTING_DEPTH
            )));
        }
        let actual_length = u64::try_from(self.values.len()).map_err(|_| {
            GGUFError::InvalidMetadata(
                "Array length does not fit the GGUF u64 length field".to_string(),
            )
        })?;
        if self.length != actual_length {
            return Err(GGUFError::InvalidMetadata(format!(
                "Array length field {} does not match {} values",
                self.length,
                self.values.len()
            )));
        }
        for value in &self.values {
            if value.value_type() != self.element_type {
                return Err(GGUFError::InvalidMetadata(format!(
                    "Array element type mismatch: expected {}, got {}",
                    self.element_type,
                    value.value_type()
                )));
            }
            value.validate_at_depth(depth)?;
        }
        Ok(())
    }

    /// Read a metadata array from a reader
    #[cfg(feature = "std")]
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let mut budget = MetadataDecodeBudget::new(GGUF_MAX_METADATA_DECODED_SIZE);
        Self::read_from_at_depth(reader, 1, &mut budget)
    }

    #[cfg(feature = "std")]
    fn read_from_at_depth<R: Read>(
        reader: &mut R,
        depth: usize,
        budget: &mut MetadataDecodeBudget,
    ) -> Result<Self> {
        if depth > GGUF_MAX_METADATA_NESTING_DEPTH {
            return Err(GGUFError::InvalidMetadata(format!(
                "Metadata array nesting exceeds limit of {}",
                GGUF_MAX_METADATA_NESTING_DEPTH
            )));
        }
        let element_type_raw = read_u32(reader)?;
        let element_type = GGUFValueType::from_u32(element_type_raw)?;
        let length = read_u64(reader)?;

        let platform_length = usize::try_from(length).map_err(|_| {
            GGUFError::InvalidMetadata("Array length does not fit usize".to_string())
        })?;
        let decoded_slots = platform_length
            .checked_mul(core::mem::size_of::<MetadataValue>())
            .ok_or_else(|| {
                GGUFError::InvalidMetadata(
                    "Decoded metadata array allocation overflows usize".to_string(),
                )
            })?;
        budget.charge(decoded_slots, "array elements")?;
        let mut values = Vec::new();
        while values.len() < platform_length {
            if values.len() == values.capacity() {
                let remaining = platform_length - values.len();
                let additional = values.capacity().max(METADATA_ARRAY_RESERVE_CHUNK).min(remaining);
                values.try_reserve_exact(additional).map_err(|_| {
                    GGUFError::InvalidMetadata(format!(
                        "Unable to allocate storage for metadata array of {} elements",
                        length
                    ))
                })?;
            }
            let batch_end = values
                .len()
                .saturating_add(
                    (values.capacity() - values.len()).min(METADATA_ARRAY_RESERVE_CHUNK),
                )
                .min(platform_length);
            while values.len() < batch_end {
                let value = MetadataValue::read_from_at_depth(reader, element_type, depth, budget)?;
                values.push(value);
            }
        }

        Ok(Self { element_type, length, values })
    }

    /// Write a metadata array to a writer
    #[cfg(feature = "std")]
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.validate()?;
        write_u32(writer, self.element_type as u32)?;
        write_u64(writer, self.length)?;

        for value in &self.values {
            value.write_to(writer)?;
        }

        Ok(())
    }

    /// Get the length of the array
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get an element by index
    pub fn get(&self, index: usize) -> Option<&MetadataValue> {
        self.values.get(index)
    }

    /// Iterate over the values
    #[cfg(feature = "std")]
    pub fn iter(&self) -> std::slice::Iter<'_, MetadataValue> {
        self.values.iter()
    }

    #[cfg(not(feature = "std"))]
    pub fn iter(&self) -> slice::Iter<'_, MetadataValue> {
        self.values.iter()
    }
}

#[cfg(feature = "std")]
fn read_variable_bytes<R: Read>(
    reader: &mut R,
    length: u64,
    kind: &str,
    budget: &mut MetadataDecodeBudget,
) -> Result<Vec<u8>> {
    let length = usize::try_from(length)
        .map_err(|_| GGUFError::Format(format!("{} length does not fit usize", kind)))?;
    budget.charge(length, kind)?;
    let mut bytes = Vec::new();
    let mut chunk = [0u8; METADATA_BYTE_READ_CHUNK];
    while bytes.len() < length {
        if bytes.len() == bytes.capacity() {
            let remaining = length - bytes.len();
            let additional = bytes.capacity().max(METADATA_BYTE_READ_CHUNK).min(remaining);
            bytes.try_reserve_exact(additional).map_err(|_| {
                GGUFError::InvalidMetadata(format!(
                    "Unable to allocate storage for metadata {} of {} bytes",
                    kind, length
                ))
            })?;
        }
        let chunk_len = (length - bytes.len()).min(chunk.len());
        reader.read_exact(&mut chunk[..chunk_len])?;
        bytes.extend_from_slice(&chunk[..chunk_len]);
    }
    Ok(bytes)
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl Metadata {
    /// Create a new empty metadata collection
    pub fn new() -> Self {
        Self { data: HashMap::new() }
    }

    /// Insert a key-value pair
    pub fn insert(&mut self, key: String, value: MetadataValue) {
        self.data.insert(key, value);
    }

    /// Get a value by key
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.data.get(key)
    }

    /// Remove a key-value pair
    pub fn remove(&mut self, key: &str) -> Option<MetadataValue> {
        self.data.remove(key)
    }

    /// Check if a key exists
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Get the number of key-value pairs
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the metadata is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over key-value pairs
    #[cfg(feature = "std")]
    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, String, MetadataValue> {
        self.data.iter()
    }

    #[cfg(all(not(feature = "std"), feature = "alloc"))]
    pub fn iter(&self) -> hashbrown::hash_map::Iter<'_, String, MetadataValue> {
        self.data.iter()
    }

    /// Get all keys
    #[cfg(feature = "std")]
    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, String, MetadataValue> {
        self.data.keys()
    }

    #[cfg(all(not(feature = "std"), feature = "alloc"))]
    pub fn keys(&self) -> hashbrown::hash_map::Keys<'_, String, MetadataValue> {
        self.data.keys()
    }

    /// Get all values
    #[cfg(feature = "std")]
    pub fn values(&self) -> std::collections::hash_map::Values<'_, String, MetadataValue> {
        self.data.values()
    }

    #[cfg(all(not(feature = "std"), feature = "alloc"))]
    pub fn values(&self) -> hashbrown::hash_map::Values<'_, String, MetadataValue> {
        self.data.values()
    }

    /// Validate metadata invariants before writing or using format-level settings.
    pub fn validate(&self) -> Result<()> {
        if self.data.len() as u64 > GGUF_MAX_METADATA_COUNT {
            return Err(GGUFError::InvalidMetadata(format!(
                "Too many metadata entries: {} exceeds limit of {}",
                self.data.len(),
                GGUF_MAX_METADATA_COUNT
            )));
        }
        for (key, value) in &self.data {
            if key.is_empty() {
                return Err(GGUFError::InvalidMetadata("Metadata key cannot be empty".to_string()));
            }
            if key.len() > GGUF_MAX_METADATA_KEY_LENGTH {
                return Err(GGUFError::InvalidMetadata(format!(
                    "Metadata key is {} bytes; maximum is {}",
                    key.len(),
                    GGUF_MAX_METADATA_KEY_LENGTH
                )));
            }
            if !is_valid_metadata_key(key) {
                return Err(GGUFError::InvalidMetadata(format!(
                    "Metadata key '{}' must be dot-separated lower_snake_case ASCII",
                    key
                )));
            }
            value.validate()?;
        }
        self.tensor_alignment()?;
        Ok(())
    }

    /// Return the global tensor alignment declared by `general.alignment`.
    pub fn tensor_alignment(&self) -> Result<usize> {
        let alignment = match self.get("general.alignment") {
            None => GGUF_DEFAULT_ALIGNMENT,
            Some(MetadataValue::U32(value)) => usize::try_from(*value).map_err(|_| {
                GGUFError::InvalidMetadata(
                    "general.alignment does not fit this platform".to_string(),
                )
            })?,
            Some(value) => {
                return Err(GGUFError::InvalidMetadata(format!(
                    "general.alignment must be u32, got {}",
                    value.value_type()
                )));
            }
        };
        if alignment == 0 || alignment % 8 != 0 {
            return Err(GGUFError::InvalidMetadata(format!(
                "general.alignment must be a non-zero multiple of 8, got {}",
                alignment
            )));
        }
        Ok(alignment)
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
fn is_valid_metadata_key(key: &str) -> bool {
    key.is_ascii()
        && key.split('.').all(|component| {
            !component.is_empty()
                && component.split('_').all(|word| {
                    !word.is_empty()
                        && word
                            .bytes()
                            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
                })
        })
}

#[cfg(not(any(feature = "std", feature = "alloc")))]
impl Metadata {
    /// Create a new empty metadata collection (no-op for no_std + no_alloc)
    pub fn new() -> Self {
        Self { data: () }
    }

    /// Insert a key-value pair (no-op for no_std + no_alloc)
    pub fn insert(&mut self, _key: String, _value: MetadataValue) {
        // No-op: can't store data without allocation
    }

    /// Get a value by key (always returns None for no_std + no_alloc)
    pub fn get(&self, _key: &str) -> Option<&MetadataValue> {
        None
    }

    /// Remove a key-value pair (always returns None for no_std + no_alloc)
    pub fn remove(&mut self, _key: &str) -> Option<MetadataValue> {
        None
    }

    /// Check if a key exists (always returns false for no_std + no_alloc)
    pub fn contains_key(&self, _key: &str) -> bool {
        false
    }

    /// Get the number of key-value pairs (always 0 for no_std + no_alloc)
    pub fn len(&self) -> usize {
        0
    }

    /// Check if the metadata is empty (always true for no_std + no_alloc)
    pub fn is_empty(&self) -> bool {
        true
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl Metadata {
    /// Read metadata from a reader
    #[cfg(feature = "std")]
    pub fn read_from<R: Read>(reader: &mut R, count: u64) -> Result<Self> {
        let mut budget = MetadataDecodeBudget::new(GGUF_MAX_METADATA_DECODED_SIZE);
        Self::read_from_with_decode_budget(reader, count, &mut budget)
    }

    #[cfg(feature = "std")]
    fn read_from_with_decode_budget<R: Read>(
        reader: &mut R,
        count: u64,
        budget: &mut MetadataDecodeBudget,
    ) -> Result<Self> {
        if count > GGUF_MAX_METADATA_COUNT {
            return Err(GGUFError::InvalidMetadata(format!(
                "Metadata count {} exceeds limit of {}",
                count, GGUF_MAX_METADATA_COUNT
            )));
        }
        let capacity = usize::try_from(count).map_err(|_| {
            GGUFError::InvalidMetadata("Metadata count does not fit usize".to_string())
        })?;
        let decoded_entries = capacity
            .checked_mul(
                core::mem::size_of::<(String, MetadataValue)>()
                    .saturating_add(2 * core::mem::size_of::<usize>()),
            )
            .ok_or_else(|| {
                GGUFError::InvalidMetadata(
                    "Decoded metadata entry allocation overflows usize".to_string(),
                )
            })?;
        budget.charge(decoded_entries, "metadata entries")?;
        let mut data = HashMap::new();
        data.try_reserve(capacity).map_err(|_| {
            GGUFError::InvalidMetadata("Unable to allocate metadata map".to_string())
        })?;

        for _ in 0..count {
            // Read key
            let key_len = read_u64(reader)?;
            if key_len > GGUF_MAX_METADATA_KEY_LENGTH as u64 {
                return Err(GGUFError::Format(format!("Metadata key too long: {} bytes", key_len)));
            }
            let key_len = usize::try_from(key_len).map_err(|_| {
                GGUFError::Format("Metadata key length does not fit usize".to_string())
            })?;
            budget.charge(key_len, "metadata key")?;

            let mut key_bytes = vec![0u8; key_len];
            reader.read_exact(&mut key_bytes)?;
            let key = String::from_utf8(key_bytes)
                .map_err(|e| GGUFError::Format(format!("Invalid UTF-8 in metadata key: {}", e)))?;
            if key.is_empty() {
                return Err(GGUFError::InvalidMetadata("Metadata key cannot be empty".to_string()));
            }
            if data.contains_key(&key) {
                return Err(GGUFError::InvalidMetadata(format!("Duplicate metadata key: {}", key)));
            }

            // Read value type and value
            let value_type_raw = read_u32(reader)?;
            let value_type = GGUFValueType::from_u32(value_type_raw)?;
            let value = MetadataValue::read_from_at_depth(reader, value_type, 0, budget)?;

            data.insert(key, value);
        }

        let metadata = Self { data };
        metadata.validate()?;
        Ok(metadata)
    }

    /// Read metadata with a serialized-byte budget and the default decoded-allocation budget.
    #[cfg(feature = "std")]
    pub fn read_from_with_limit<R: Read>(
        reader: &mut R,
        count: u64,
        max_bytes: usize,
    ) -> Result<Self> {
        Self::read_from_with_limits(reader, count, max_bytes, GGUF_MAX_METADATA_DECODED_SIZE)
    }

    /// Read metadata with independent serialized-byte and decoded-allocation budgets.
    #[cfg(feature = "std")]
    pub fn read_from_with_limits<R: Read>(
        reader: &mut R,
        count: u64,
        max_bytes: usize,
        max_decoded_bytes: usize,
    ) -> Result<Self> {
        let limit = u64::try_from(max_bytes).unwrap_or(u64::MAX);
        let mut limited = reader.take(limit);
        let mut budget = MetadataDecodeBudget::new(max_decoded_bytes);
        match Self::read_from_with_decode_budget(&mut limited, count, &mut budget) {
            Ok(metadata) => Ok(metadata),
            Err(_) if limited.limit() == 0 => Err(GGUFError::InvalidMetadata(format!(
                "Metadata exceeds byte limit of {}",
                max_bytes
            ))),
            Err(error) => Err(error),
        }
    }

    /// Write metadata to a writer
    #[cfg(feature = "std")]
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.validate()?;
        let mut entries = Vec::new();
        entries.try_reserve_exact(self.data.len()).map_err(|_| {
            GGUFError::InvalidMetadata(format!(
                "Unable to allocate deterministic metadata index for {} entries",
                self.data.len()
            ))
        })?;
        entries.extend(self.data.iter());
        entries.sort_unstable_by_key(|(key, _)| *key);
        for (key, value) in entries {
            // Write key
            let key_len = u64::try_from(key.len()).map_err(|_| {
                GGUFError::InvalidMetadata(format!(
                    "Metadata key length does not fit the GGUF u64 length field: {} bytes",
                    key.len()
                ))
            })?;
            write_u64(writer, key_len)?;
            writer.write_all(key.as_bytes())?;

            // Write value type and value
            write_u32(writer, value.value_type() as u32)?;
            value.write_to(writer)?;
        }

        Ok(())
    }

    /// Calculate the serialized size of all metadata
    pub fn serialized_size(&self) -> usize {
        self.checked_serialized_size().unwrap_or(usize::MAX)
    }

    /// Calculate the serialized metadata size without overflowing.
    pub fn checked_serialized_size(&self) -> Option<usize> {
        self.data.iter().try_fold(0usize, |size, (key, value)| {
            size.checked_add(8)?
                .checked_add(key.len())?
                .checked_add(4)?
                .checked_add(value.checked_serialized_size()?)
        })
    }

    /// Get a string value by key
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    /// Get a u64 value by key
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }

    /// Get an i64 value by key
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.get(key)?.as_i64()
    }

    /// Get an f64 value by key
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key)?.as_f64()
    }

    /// Get a bool value by key
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_representation())
    }
}

#[cfg(not(feature = "std"))]
impl fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetadataValue::U8(v) => write!(f, "{}", v),
            MetadataValue::I8(v) => write!(f, "{}", v),
            MetadataValue::U16(v) => write!(f, "{}", v),
            MetadataValue::I16(v) => write!(f, "{}", v),
            MetadataValue::U32(v) => write!(f, "{}", v),
            MetadataValue::I32(v) => write!(f, "{}", v),
            MetadataValue::F32(v) => write!(f, "{}", v),
            MetadataValue::Bool(v) => write!(f, "{}", v),
            MetadataValue::String(v) => write!(f, "\"{}\"", v),
            MetadataValue::Array(arr) => {
                write!(f, "[")?;
                for (i, value) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", value)?;
                }
                write!(f, "]")
            }
            MetadataValue::U64(v) => write!(f, "{}", v),
            MetadataValue::I64(v) => write!(f, "{}", v),
            MetadataValue::F64(v) => write!(f, "{}", v),
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_metadata_value_types() {
        let val = MetadataValue::U32(42);
        assert_eq!(val.value_type(), GGUFValueType::U32);
        assert_eq!(val.as_u64(), Some(42));

        let val = MetadataValue::String("hello".to_string());
        assert_eq!(val.value_type(), GGUFValueType::String);
        assert_eq!(val.as_str(), Some("hello"));

        let val = MetadataValue::Bool(true);
        assert_eq!(val.value_type(), GGUFValueType::Bool);
        assert_eq!(val.as_bool(), Some(true));
    }

    #[test]
    fn test_metadata_value_io() {
        let original = MetadataValue::F32(std::f32::consts::PI);

        let mut buffer = Vec::new();
        original.write_to(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_value = MetadataValue::read_from(&mut cursor, GGUFValueType::F32).unwrap();

        assert_eq!(original, read_value);
    }

    #[test]
    fn test_metadata_array() {
        let values = vec![MetadataValue::U32(1), MetadataValue::U32(2), MetadataValue::U32(3)];

        let array = MetadataArray::new(GGUFValueType::U32, values).unwrap();
        assert_eq!(array.len(), 3);
        assert_eq!(array.element_type, GGUFValueType::U32);
        assert_eq!(array.get(0), Some(&MetadataValue::U32(1)));
    }

    #[test]
    fn test_metadata_array_type_mismatch() {
        let values = vec![
            MetadataValue::U32(1),
            MetadataValue::String("hello".to_string()), // Wrong type!
        ];

        let result = MetadataArray::new(GGUFValueType::U32, values);
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_collection() {
        let mut metadata = Metadata::new();
        metadata.insert("name".to_string(), MetadataValue::String("test_model".to_string()));
        metadata.insert("version".to_string(), MetadataValue::U32(1));

        assert_eq!(metadata.len(), 2);
        assert_eq!(metadata.get_string("name"), Some("test_model"));
        assert_eq!(metadata.get_u64("version"), Some(1));
        assert!(metadata.contains_key("name"));
        assert!(!metadata.contains_key("nonexistent"));
    }

    #[test]
    fn test_metadata_io() {
        let mut original = Metadata::new();
        original.insert("name".to_string(), MetadataValue::String("test".to_string()));
        original.insert("count".to_string(), MetadataValue::U64(42));

        let mut buffer = Vec::new();
        original.write_to(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_metadata = Metadata::read_from(&mut cursor, 2).unwrap();

        assert_eq!(read_metadata.len(), 2);
        assert_eq!(read_metadata.get_string("name"), Some("test"));
        assert_eq!(read_metadata.get_u64("count"), Some(42));
    }

    #[test]
    fn test_large_string_round_trip() {
        let long_value = MetadataValue::String("x".repeat(128 * 1024));
        let mut buffer = Vec::new();
        long_value.write_to(&mut buffer).unwrap();

        let decoded =
            MetadataValue::read_from(&mut Cursor::new(buffer), GGUFValueType::String).unwrap();
        assert_eq!(decoded, long_value);
    }

    #[test]
    fn test_array_io() {
        let values = vec![MetadataValue::I32(-1), MetadataValue::I32(0), MetadataValue::I32(42)];

        let original = MetadataArray::new(GGUFValueType::I32, values).unwrap();

        let mut buffer = Vec::new();
        original.write_to(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_array = MetadataArray::read_from(&mut cursor).unwrap();

        assert_eq!(read_array.element_type, GGUFValueType::I32);
        assert_eq!(read_array.len(), 3);
        assert_eq!(read_array.get(2), Some(&MetadataValue::I32(42)));
    }

    #[test]
    fn test_large_tokenizer_array_round_trip() {
        let values = (0..128_000).map(MetadataValue::U32).collect();
        let original = MetadataArray::new(GGUFValueType::U32, values).unwrap();

        let mut buffer = Vec::new();
        original.write_to(&mut buffer).unwrap();
        let decoded = MetadataArray::read_from(&mut Cursor::new(buffer)).unwrap();

        assert_eq!(decoded, original);
    }

    #[test]
    fn test_truncated_huge_string_declaration_is_rejected_without_large_allocation() {
        let encoded_length = u64::MAX.to_le_bytes();
        let error =
            MetadataValue::read_from(&mut Cursor::new(encoded_length), GGUFValueType::String)
                .unwrap_err();

        assert!(matches!(
            error,
            GGUFError::Io(_) | GGUFError::Format(_) | GGUFError::InvalidMetadata(_)
        ));
    }

    #[test]
    fn test_truncated_huge_array_declaration_is_rejected_without_large_allocation() {
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&(GGUFValueType::U8 as u32).to_le_bytes());
        encoded.extend_from_slice(&u64::MAX.to_le_bytes());

        let error = MetadataArray::read_from(&mut Cursor::new(encoded)).unwrap_err();
        assert!(matches!(error, GGUFError::Io(_) | GGUFError::InvalidMetadata(_)));
    }

    #[test]
    fn test_compact_array_is_rejected_before_decoded_memory_amplification() {
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&6u64.to_le_bytes());
        encoded.extend_from_slice(b"tokens");
        encoded.extend_from_slice(&(GGUFValueType::Array as u32).to_le_bytes());
        encoded.extend_from_slice(&(GGUFValueType::U8 as u32).to_le_bytes());
        let oversized_elements =
            GGUF_MAX_METADATA_DECODED_SIZE / core::mem::size_of::<MetadataValue>() + 1;
        encoded.extend_from_slice(&(oversized_elements as u64).to_le_bytes());

        let error =
            Metadata::read_from_with_limit(&mut Cursor::new(encoded), 1, GGUF_MAX_METADATA_SIZE)
                .unwrap_err();
        assert!(error.to_string().contains("Decoded metadata allocation exceeds budget"));
    }

    #[test]
    fn test_metadata_key_validation() {
        for key in ["general.name", "tokenizer.ggml.model", "block_count", "layer.0.name"] {
            let mut metadata = Metadata::new();
            metadata.insert(key.to_string(), MetadataValue::U32(1));
            assert!(metadata.validate().is_ok(), "valid key rejected: {key}");
        }

        for key in [
            "Uppercase",
            "has-dash",
            ".leading",
            "trailing.",
            "double..dot",
            "_leading",
            "trailing_",
            "double__underscore",
        ] {
            let mut metadata = Metadata::new();
            metadata.insert(key.to_string(), MetadataValue::U32(1));
            assert!(metadata.validate().is_err(), "invalid key accepted: {key}");
        }
    }

    #[test]
    fn test_invalid_bool_is_rejected() {
        let mut cursor = Cursor::new([2u8]);
        assert!(MetadataValue::read_from(&mut cursor, GGUFValueType::Bool).is_err());
    }

    #[test]
    fn test_array_length_mismatch_is_rejected() {
        let array = MetadataArray {
            element_type: GGUFValueType::U32,
            length: 2,
            values: vec![MetadataValue::U32(1)],
        };
        assert!(array.validate().is_err());
        assert!(array.write_to(&mut Vec::new()).is_err());
    }
}
