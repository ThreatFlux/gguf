//! Memory mapping support for GGUF files
//!
//! This module provides memory-mapped I/O for efficient access to large GGUF files.

#[cfg(feature = "mmap")]
use crate::{
    error::{GGUFError, Result},
    format::constants::{GGUF_MAGIC, GGUF_VERSION},
};

#[cfg(feature = "mmap")]
use memmap2::Mmap;
#[cfg(feature = "mmap")]
use std::{fs::File, path::Path, sync::Arc};

/// Header-only view of a memory-mapped GGUF file.
///
/// This preview API validates the GGUF magic and version prefix and retains the
/// underlying mapping. It does not parse metadata or tensor descriptors.
#[cfg(feature = "mmap")]
pub struct MmapGGUFFile {
    mmap: Arc<Mmap>,
    version: u32,
}

#[cfg(feature = "mmap")]
impl MmapGGUFFile {
    /// Memory-map a GGUF file and validate its magic and version prefix.
    ///
    /// This does not perform a full GGUF parse.
    ///
    /// # Safety
    ///
    /// The mapped file must not be truncated or modified for the lifetime of
    /// the returned value. This crate cannot enforce that condition against
    /// other processes or independently opened file handles.
    pub unsafe fn mmap<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: The caller accepts the file-stability contract documented
        // above. `MmapGGUFFile` retains the mapping for its full lifetime.
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

        Self::from_mmap(mmap)
    }

    /// Create a header-only GGUF view from an existing memory map.
    ///
    /// This validates only the eight-byte magic and version prefix.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Result<Self> {
        let header = mmap.get(..8).ok_or(GGUFError::UnexpectedEof)?;
        let magic_bytes: [u8; 4] = header
            .get(0..4)
            .ok_or(GGUFError::UnexpectedEof)?
            .try_into()
            .map_err(|_| GGUFError::UnexpectedEof)?;
        let magic = u32::from_le_bytes(magic_bytes);

        if magic != GGUF_MAGIC {
            return Err(GGUFError::InvalidMagic { expected: GGUF_MAGIC, found: magic });
        }

        let version_bytes: [u8; 4] = header
            .get(4..8)
            .ok_or(GGUFError::UnexpectedEof)?
            .try_into()
            .map_err(|_| GGUFError::UnexpectedEof)?;
        let version = u32::from_le_bytes(version_bytes);

        if version != GGUF_VERSION {
            return Err(GGUFError::UnsupportedVersion(version));
        }

        Ok(Self { mmap, version })
    }

    /// Return the validated GGUF version.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Return the mapped file length in bytes.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Return whether the mapped file is empty.
    ///
    /// A successfully constructed value is never empty because prefix
    /// validation requires at least eight bytes.
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Borrow the complete mapped file as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Create a bounded primitive reader over the mapped file.
    pub fn reader(&self) -> MmapGGUFReader {
        MmapGGUFReader::new(Arc::clone(&self.mmap))
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
        let start = self.position;
        let end = start.checked_add(count).ok_or(GGUFError::UnexpectedEof)?;
        let bytes = self.mmap.get(start..end).ok_or(GGUFError::UnexpectedEof)?;
        self.position = end;

        Ok(bytes)
    }

    /// Read a u32 value in little-endian format
    pub fn read_u32(&mut self) -> Result<u32> {
        let bytes: [u8; 4] =
            self.read_bytes(4)?.try_into().map_err(|_| GGUFError::UnexpectedEof)?;
        Ok(u32::from_le_bytes(bytes))
    }

    /// Read a u64 value in little-endian format
    pub fn read_u64(&mut self) -> Result<u64> {
        let bytes: [u8; 8] =
            self.read_bytes(8)?.try_into().map_err(|_| GGUFError::UnexpectedEof)?;
        Ok(u64::from_le_bytes(bytes))
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

        let result = unsafe { MmapGGUFFile::mmap(temp_file.path()) };
        assert!(matches!(result, Err(GGUFError::InvalidMagic { .. })));
    }

    #[test]
    fn test_mmap_rejects_truncated_header() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        temp_file.flush().unwrap();

        let result = unsafe { MmapGGUFFile::mmap(temp_file.path()) };
        assert!(matches!(result, Err(GGUFError::UnexpectedEof)));
    }

    #[test]
    fn test_mmap_valid_magic_invalid_version() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap(); // Valid magic
        temp_file.write_all(&999u32.to_le_bytes()).unwrap(); // Invalid version
        temp_file.flush().unwrap();

        let result = unsafe { MmapGGUFFile::mmap(temp_file.path()) };
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

    #[test]
    fn test_mmap_file_exposes_validated_mapping() {
        let data = [0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00];
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&data).unwrap();
        temp_file.flush().unwrap();

        let mapped = unsafe { MmapGGUFFile::mmap(temp_file.path()) }.unwrap();
        assert_eq!(mapped.version(), GGUF_VERSION);
        assert_eq!(mapped.len(), data.len());
        assert!(!mapped.is_empty());
        assert_eq!(mapped.as_bytes(), data);

        let mut reader = mapped.reader();
        assert_eq!(reader.read_u32().unwrap(), GGUF_MAGIC);
    }

    #[test]
    fn test_mmap_reader_rejects_overflow_without_advancing() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&[0; 8]).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let mut reader = MmapGGUFReader::new(mmap);
        reader.seek(1).unwrap();

        assert!(matches!(reader.read_bytes(usize::MAX), Err(GGUFError::UnexpectedEof)));
        assert_eq!(reader.position(), 1);
    }

    #[test]
    fn test_mmap_reader_rejects_out_of_range_read_without_advancing() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&[0; 8]).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let mut reader = MmapGGUFReader::new(mmap);
        reader.seek(7).unwrap();

        assert!(matches!(reader.read_u32(), Err(GGUFError::UnexpectedEof)));
        assert_eq!(reader.position(), 7);
    }
}

#[cfg(not(feature = "mmap"))]
compile_error!("This module requires the 'mmap' feature to be enabled");
