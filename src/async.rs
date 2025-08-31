//! Async I/O support for GGUF files
//!
//! This module provides async variants of GGUF operations using Tokio.

#[cfg(feature = "async")]
use crate::{
    error::GGUFError,
    error::Result,
    format::constants::{GGUF_MAGIC, GGUF_VERSION},
    format::Metadata,
};

#[cfg(feature = "async")]
use tokio::io::{AsyncRead, AsyncReadExt};

/// Async GGUF file operations
#[cfg(feature = "async")]
pub struct AsyncGGUFFile {
    pub version: u32,
    pub metadata: Metadata,
    pub tensors: Vec<crate::tensor::TensorInfo>,
}

#[cfg(feature = "async")]
impl AsyncGGUFFile {
    /// Async version of reading a GGUF file
    pub async fn read_async<R: AsyncRead + Unpin>(mut reader: R) -> Result<Self> {
        // Read magic number
        let mut magic_bytes = [0u8; 4];
        reader.read_exact(&mut magic_bytes).await?;
        let magic = u32::from_le_bytes(magic_bytes);

        if magic != GGUF_MAGIC {
            return Err(GGUFError::InvalidMagic { expected: GGUF_MAGIC, found: magic });
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes).await?;
        let version = u32::from_le_bytes(version_bytes);

        if version != GGUF_VERSION {
            return Err(GGUFError::UnsupportedVersion(version));
        }

        // TODO: Implement full async GGUF parsing
        // This is a stub implementation that will be expanded

        Ok(Self { version, metadata: Metadata::new(), tensors: Vec::new() })
    }

    /// Read a GGUF file from a file path asynchronously
    pub async fn read_file_async<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let file = tokio::fs::File::open(path).await?;
        Self::read_async(file).await
    }
}

#[cfg(all(feature = "async", test))]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_async_invalid_magic() {
        let data = [0x00, 0x00, 0x00, 0x00]; // Invalid magic
        let reader = Cursor::new(data);

        let result = AsyncGGUFFile::read_async(reader).await;
        assert!(matches!(result, Err(GGUFError::InvalidMagic { .. })));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_async_valid_magic_invalid_version() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes()); // Valid magic
        data.extend_from_slice(&999u32.to_le_bytes()); // Invalid version

        let reader = Cursor::new(data);
        let result = AsyncGGUFFile::read_async(reader).await;
        assert!(matches!(result, Err(GGUFError::UnsupportedVersion(999))));
    }
}

#[cfg(not(feature = "async"))]
compile_error!("This module requires the 'async' feature to be enabled");
