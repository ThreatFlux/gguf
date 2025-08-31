//! Simple little-endian byte operations to avoid byteorder recursion issues

use crate::error::{GGUFError, Result};
use std::io::{Read, Write};

/// Read a u32 in little-endian format
pub fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).map_err(GGUFError::from)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a u64 in little-endian format
pub fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).map_err(GGUFError::from)?;
    Ok(u64::from_le_bytes(buf))
}

/// Read an f32 in little-endian format
pub fn read_f32<R: Read>(reader: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).map_err(GGUFError::from)?;
    Ok(f32::from_le_bytes(buf))
}

/// Read an f64 in little-endian format
pub fn read_f64<R: Read>(reader: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).map_err(GGUFError::from)?;
    Ok(f64::from_le_bytes(buf))
}

/// Read a u16 in little-endian format
pub fn read_u16<R: Read>(reader: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf).map_err(GGUFError::from)?;
    Ok(u16::from_le_bytes(buf))
}

/// Read an i16 in little-endian format
pub fn read_i16<R: Read>(reader: &mut R) -> Result<i16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf).map_err(GGUFError::from)?;
    Ok(i16::from_le_bytes(buf))
}

/// Read an i32 in little-endian format
pub fn read_i32<R: Read>(reader: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).map_err(GGUFError::from)?;
    Ok(i32::from_le_bytes(buf))
}

/// Read an i64 in little-endian format
pub fn read_i64<R: Read>(reader: &mut R) -> Result<i64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).map_err(GGUFError::from)?;
    Ok(i64::from_le_bytes(buf))
}

/// Read a u8
pub fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf).map_err(GGUFError::from)?;
    Ok(buf[0])
}

/// Read an i8
pub fn read_i8<R: Read>(reader: &mut R) -> Result<i8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf).map_err(GGUFError::from)?;
    Ok(buf[0] as i8)
}

/// Write a u32 in little-endian format
pub fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes()).map_err(GGUFError::from)
}

/// Write a u64 in little-endian format
pub fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes()).map_err(GGUFError::from)
}

/// Write an f32 in little-endian format
pub fn write_f32<W: Write>(writer: &mut W, value: f32) -> Result<()> {
    writer.write_all(&value.to_le_bytes()).map_err(GGUFError::from)
}

/// Write an f64 in little-endian format
pub fn write_f64<W: Write>(writer: &mut W, value: f64) -> Result<()> {
    writer.write_all(&value.to_le_bytes()).map_err(GGUFError::from)
}

/// Write a u16 in little-endian format
pub fn write_u16<W: Write>(writer: &mut W, value: u16) -> Result<()> {
    writer.write_all(&value.to_le_bytes()).map_err(GGUFError::from)
}

/// Write an i16 in little-endian format
pub fn write_i16<W: Write>(writer: &mut W, value: i16) -> Result<()> {
    writer.write_all(&value.to_le_bytes()).map_err(GGUFError::from)
}

/// Write an i32 in little-endian format
pub fn write_i32<W: Write>(writer: &mut W, value: i32) -> Result<()> {
    writer.write_all(&value.to_le_bytes()).map_err(GGUFError::from)
}

/// Write an i64 in little-endian format
pub fn write_i64<W: Write>(writer: &mut W, value: i64) -> Result<()> {
    writer.write_all(&value.to_le_bytes()).map_err(GGUFError::from)
}

/// Write a u8
pub fn write_u8<W: Write>(writer: &mut W, value: u8) -> Result<()> {
    writer.write_all(&[value]).map_err(GGUFError::from)
}

/// Write an i8
pub fn write_i8<W: Write>(writer: &mut W, value: i8) -> Result<()> {
    writer.write_all(&[value as u8]).map_err(GGUFError::from)
}
