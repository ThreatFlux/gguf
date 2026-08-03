//! GGUF format constants and magic numbers

/// Magic number for GGUF files ("GGUF" in little-endian byte order)
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Current GGUF format version
pub const GGUF_VERSION: u32 = 3;

/// Current GGML quantization format version written for quantized tensors.
pub const GGUF_QUANTIZATION_VERSION: u32 = 2;

/// Default alignment requirement for tensor data (32 bytes)
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// Maximum metadata key length allowed by the GGUF specification.
pub const GGUF_MAX_METADATA_KEY_LENGTH: usize = 65_535;

/// Maximum tensor name length allowed by the GGUF specification.
pub const GGUF_MAX_TENSOR_NAME_LENGTH: usize = 64;

/// Maximum tensor rank supported by GGUF v3.
pub const GGUF_MAX_DIMENSIONS: usize = 4;

/// Maximum supported nesting depth for metadata arrays.
pub const GGUF_MAX_METADATA_NESTING_DEPTH: usize = 64;

/// Maximum number of metadata entries accepted by the parser.
pub const GGUF_MAX_METADATA_COUNT: u64 = 100_000;

/// Maximum number of tensors accepted by the parser.
pub const GGUF_MAX_TENSOR_COUNT: u64 = 1_000_000;

/// Default aggregate serialized metadata byte budget for seekable readers.
pub const GGUF_MAX_METADATA_SIZE: usize = 256 * 1024 * 1024;

/// Default aggregate decoded metadata allocation budget.
///
/// This separately bounds in-memory amplification from compact encodings such
/// as arrays of one-byte values.
pub const GGUF_MAX_METADATA_DECODED_SIZE: usize = 256 * 1024 * 1024;

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
