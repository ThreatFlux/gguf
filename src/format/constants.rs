//! GGUF format constants and magic numbers

/// Magic number for GGUF files ("GGUF" in little-endian byte order)
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Current GGUF format version
pub const GGUF_VERSION: u32 = 3;

/// Default alignment requirement for tensor data (32 bytes)
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// Maximum supported string length in metadata
pub const GGUF_MAX_STRING_LENGTH: usize = 65_536;

/// Maximum supported array length in metadata
pub const GGUF_MAX_ARRAY_LENGTH: usize = 65_536;

/// Size of the GGUF header in bytes (magic + version + tensor_count + metadata_kv_count)
pub const GGUF_HEADER_SIZE: usize = 4 + 4 + 8 + 8;

/// Minimum valid GGUF file size (header only)
pub const GGUF_MIN_FILE_SIZE: usize = GGUF_HEADER_SIZE;

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_magic_number() {
        // Verify the magic number is "GGUF" in little-endian
        let magic_bytes = GGUF_MAGIC.to_le_bytes();
        assert_eq!(magic_bytes, [b'G', b'G', b'U', b'F']);
    }

    #[test]
    fn test_constants() {
        assert_eq!(GGUF_VERSION, 3);
        assert_eq!(GGUF_DEFAULT_ALIGNMENT, 32);
        assert_eq!(GGUF_HEADER_SIZE, 24);
    }
}
