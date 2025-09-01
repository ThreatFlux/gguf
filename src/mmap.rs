//! Memory mapping support for GGUF files
//!
//! This module provides memory-mapped I/O for efficient access to large GGUF files.

#[cfg(feature = "mmap")]
use crate::{
    error::{GGUFError, Result},
    format::constants::{GGUF_MAGIC, GGUF_VERSION},
    metadata::Metadata,
};

#[cfg(feature = "mmap")]
use memmap2::Mmap;
#[cfg(feature = "mmap")]
use std::{fs::File, path::Path, sync::Arc};

/// Memory-mapped GGUF file
#[cfg(feature = "mmap")]
pub struct MmapGGUFFile {
    #[allow(dead_code)]
    mmap: Arc<Mmap>,
    #[allow(dead_code)]
    version: u32,
    #[allow(dead_code)]
    metadata: Metadata,
}

#[cfg(feature = "mmap")]
impl MmapGGUFFile {
    /// Memory-map a GGUF file for efficient access
    pub fn mmap<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

        Self::from_mmap(mmap)
    }

    /// Create a GGUF file from an existing memory map
    pub fn from_mmap(mmap: Arc<Mmap>) -> Result<Self> {
        // Check minimum file size
        if mmap.len() < 8 {
            return Err(GGUFError::UnexpectedEof);
        }

        // Read magic number
        let magic_bytes = [mmap[0], mmap[1], mmap[2], mmap[3]];
        let magic = u32::from_le_bytes(magic_bytes);

        if magic != GGUF_MAGIC {
            return Err(GGUFError::InvalidMagic { expected: GGUF_MAGIC, found: magic });
        }

        // Read version
        let version_bytes = [mmap[4], mmap[5], mmap[6], mmap[7]];
        let version = u32::from_le_bytes(version_bytes);

        if version != GGUF_VERSION {
            return Err(GGUFError::UnsupportedVersion(version));
        }

        // TODO: Implement full memory-mapped GGUF parsing
        // This is a stub implementation that will be expanded

        Ok(Self { mmap, version, metadata: Metadata::new() })
    }
}

/// Memory-mapped GGUF reader for streaming access
#[cfg(feature = "mmap")]
pub struct MmapGGUFReader {
    mmap: Arc<Mmap>,
    position: usize,
}

#[cfg(feature = "mmap")]
impl MmapGGUFReader {
    /// Create a new memory-mapped GGUF reader
    pub fn new(mmap: Arc<Mmap>) -> Self {
        Self { mmap, position: 0 }
    }

    /// Get the current position in the file
    pub fn position(&self) -> usize {
        self.position
    }

    /// Seek to a specific position in the file
    pub fn seek(&mut self, position: usize) -> Result<()> {
        if position > self.mmap.len() {
            return Err(GGUFError::UnexpectedEof);
        }
        self.position = position;
        Ok(())
    }

    /// Read bytes at the current position
    pub fn read_bytes(&mut self, count: usize) -> Result<&[u8]> {
        if self.position + count > self.mmap.len() {
            return Err(GGUFError::UnexpectedEof);
        }

        let start = self.position;
        let end = self.position + count;
        self.position = end;

        Ok(&self.mmap[start..end])
    }

    /// Read a u32 value in little-endian format
    pub fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read a u64 value in little-endian format
    pub fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }
}

#[cfg(all(feature = "mmap", test))]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mmap_invalid_magic() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&[0x00, 0x00, 0x00, 0x00]).unwrap(); // Invalid magic
        temp_file.write_all(&[0x00, 0x00, 0x00, 0x00]).unwrap(); // Add version bytes
        temp_file.flush().unwrap();

        let result = MmapGGUFFile::mmap(temp_file.path());
        assert!(matches!(result, Err(GGUFError::InvalidMagic { .. })));
    }

    #[test]
    fn test_mmap_valid_magic_invalid_version() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap(); // Valid magic
        temp_file.write_all(&999u32.to_le_bytes()).unwrap(); // Invalid version
        temp_file.flush().unwrap();

        let result = MmapGGUFFile::mmap(temp_file.path());
        assert!(matches!(result, Err(GGUFError::UnsupportedVersion(999))));
    }

    #[test]
    fn test_mmap_reader() {
        let data = vec![0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00]; // GGUF magic + version 3
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&data).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        let mmap = Arc::new(mmap);

        let mut reader = MmapGGUFReader::new(mmap);
        assert_eq!(reader.position(), 0);

        let magic = reader.read_u32().unwrap();
        assert_eq!(magic, GGUF_MAGIC);
        assert_eq!(reader.position(), 4);

        let version = reader.read_u32().unwrap();
        assert_eq!(version, 3);
        assert_eq!(reader.position(), 8);

        // temp_file automatically cleans up when dropped
    }
}

#[cfg(not(feature = "mmap"))]
compile_error!("This module requires the 'mmap' feature to be enabled");
