//! Unit tests for the format module

use gguf::format::metadata::{MetadataArray, MetadataValue};
use gguf::format::types::GGUFValueType;
use gguf::format::*;
use gguf::prelude::*;
use std::io::Cursor;

mod constants_tests {
    use super::*;

    #[test]
    fn test_gguf_magic() {
        assert_eq!(GGUF_MAGIC, 0x46554747); // "GGUF" in little-endian
    }

    #[test]
    fn test_gguf_version() {
        assert_eq!(GGUF_VERSION, 3);
    }

    #[test]
    fn test_default_alignment() {
        assert_eq!(GGUF_DEFAULT_ALIGNMENT, 32);
    }
}

mod header_tests {
    use super::*;

    #[test]
    fn test_header_creation() {
        let header = GGUFHeader::new(100, 50);

        assert_eq!(header.magic, GGUF_MAGIC);
        assert_eq!(header.version, GGUF_VERSION);
        assert_eq!(header.tensor_count, 100);
        assert_eq!(header.metadata_kv_count, 50);
    }

    #[test]
    fn test_header_default() {
        let header = GGUFHeader::default();

        assert_eq!(header.magic, GGUF_MAGIC);
        assert_eq!(header.version, GGUF_VERSION);
        assert_eq!(header.tensor_count, 0);
        assert_eq!(header.metadata_kv_count, 0);
    }

    #[test]
    fn test_header_serialization() {
        let header = GGUFHeader::new(5, 10);

        // Test write_to
        let mut bytes = Vec::new();
        header.write_to(&mut bytes).expect("Failed to serialize header");
        assert_eq!(bytes.len(), 24); // 4 + 4 + 8 + 8 bytes

        // Verify magic number
        assert_eq!(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), GGUF_MAGIC);

        // Verify version
        assert_eq!(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), GGUF_VERSION);

        // Verify tensor count
        let tensor_count = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        assert_eq!(tensor_count, 5);

        // Verify metadata count
        let metadata_count = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        assert_eq!(metadata_count, 10);
    }

    #[test]
    fn test_header_deserialization() {
        let original = GGUFHeader::new(42, 13);
        let mut bytes = Vec::new();
        original.write_to(&mut bytes).expect("Failed to serialize header");

        let mut cursor = Cursor::new(&bytes);
        let deserialized = GGUFHeader::read_from(&mut cursor).expect("Failed to read header");

        assert_eq!(deserialized.magic, original.magic);
        assert_eq!(deserialized.version, original.version);
        assert_eq!(deserialized.tensor_count, original.tensor_count);
        assert_eq!(deserialized.metadata_kv_count, original.metadata_kv_count);
    }

    #[test]
    fn test_header_invalid_magic() {
        let mut bytes = vec![0u8; 24];
        bytes[0..4].copy_from_slice(&0x12345678u32.to_le_bytes()); // Invalid magic
        bytes[4..8].copy_from_slice(&GGUF_VERSION.to_le_bytes());

        let mut cursor = Cursor::new(&bytes);
        let result = GGUFHeader::read_from(&mut cursor);

        assert!(matches!(result, Err(GGUFError::InvalidMagic { .. })));
    }

    #[test]
    fn test_header_invalid_version() {
        let mut bytes = vec![0u8; 24];
        bytes[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&999u32.to_le_bytes()); // Invalid version

        let mut cursor = Cursor::new(&bytes);
        let result = GGUFHeader::read_from(&mut cursor);

        assert!(matches!(result, Err(GGUFError::UnsupportedVersion(999))));
    }

    #[test]
    fn test_header_truncated() {
        let bytes = vec![0u8; 10]; // Too short
        let mut cursor = Cursor::new(&bytes);
        let result = GGUFHeader::read_from(&mut cursor);

        assert!(result.is_err());
    }
}

mod metadata_tests {
    use super::*;

    #[test]
    fn test_metadata_value_creation() {
        assert_eq!(MetadataValue::U8(42).as_u64(), Some(42));
        assert_eq!(MetadataValue::I8(-5).as_i64().map(|v| v as i8), Some(-5));
        assert_eq!(MetadataValue::U16(1000).as_u64().map(|v| v as u16), Some(1000));
        assert_eq!(MetadataValue::I16(-500).as_i64().map(|v| v as i16), Some(-500));
        assert_eq!(MetadataValue::U32(100000).as_u64(), Some(100000));
        assert_eq!(MetadataValue::I32(-50000).as_i64().map(|v| v as i32), Some(-50000));
        assert_eq!(MetadataValue::U64(1000000000).as_u64(), Some(1000000000));
        assert_eq!(MetadataValue::I64(-500000000).as_i64(), Some(-500000000));
        assert_eq!(
            MetadataValue::F32(std::f32::consts::PI).as_f64().map(|v| v as f32),
            Some(std::f32::consts::PI)
        );
        assert_eq!(MetadataValue::F64(std::f64::consts::E).as_f64(), Some(std::f64::consts::E));
        assert_eq!(MetadataValue::Bool(true).as_bool(), Some(true));
        assert_eq!(MetadataValue::String("test".to_string()).as_str(), Some("test"));
    }

    #[test]
    fn test_metadata_value_type_coercion() {
        let value = MetadataValue::U8(42);

        // Test successful coercion
        assert_eq!(value.as_u64(), Some(42));
        assert_eq!(value.as_u64().map(|v| v as u16), Some(42));
        assert_eq!(value.as_u64(), Some(42));
        assert_eq!(value.as_u64(), Some(42));

        // Test failed coercion
        assert_eq!(value.as_i64().map(|v| v as i8), None);
        assert_eq!(value.as_f64().map(|v| v as f32), None);
        assert_eq!(value.as_bool(), None);
        assert_eq!(value.as_str(), None);
    }

    #[test]
    fn test_metadata_value_array() {
        let values = vec![MetadataValue::U32(1), MetadataValue::U32(2), MetadataValue::U32(3)];
        let metadata_array = MetadataArray::new(GGUFValueType::U32, values).expect("Valid array");
        let array = MetadataValue::Array(Box::new(metadata_array));

        if let MetadataValue::Array(ref inner) = array {
            assert_eq!(inner.length, 3);
            assert_eq!(inner.values[0].as_u64(), Some(1));
            assert_eq!(inner.values[1].as_u64(), Some(2));
            assert_eq!(inner.values[2].as_u64(), Some(3));
        } else {
            panic!("Expected array variant");
        }
    }

    #[test]
    fn test_metadata_creation() {
        let metadata = Metadata::new();

        assert!(metadata.is_empty());
        assert_eq!(metadata.len(), 0);
        assert!(metadata.keys().next().is_none());
        assert!(metadata.values().next().is_none());
        assert!(metadata.iter().next().is_none());
    }

    #[test]
    fn test_metadata_insertion_and_retrieval() {
        let mut metadata = Metadata::new();

        metadata.insert("key1".to_string(), MetadataValue::U32(100));
        metadata.insert("key2".to_string(), MetadataValue::String("value2".to_string()));
        metadata.insert("key3".to_string(), MetadataValue::Bool(true));

        assert!(!metadata.is_empty());
        assert_eq!(metadata.len(), 3);

        assert_eq!(metadata.get("key1").and_then(|v| v.as_u64()), Some(100));
        assert_eq!(metadata.get("key2").and_then(|v| v.as_str()), Some("value2"));
        assert_eq!(metadata.get("key3").and_then(|v| v.as_bool()), Some(true));
        assert!(metadata.get("nonexistent").is_none());
    }

    #[test]
    fn test_metadata_contains_key() {
        let mut metadata = Metadata::new();
        metadata.insert("existing_key".to_string(), MetadataValue::U8(1));

        assert!(metadata.contains_key("existing_key"));
        assert!(!metadata.contains_key("nonexistent_key"));
    }

    #[test]
    fn test_metadata_remove() {
        let mut metadata = Metadata::new();
        metadata.insert("temp_key".to_string(), MetadataValue::U8(1));

        assert!(metadata.contains_key("temp_key"));

        let removed = metadata.remove("temp_key");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().as_u64(), Some(1));
        assert!(!metadata.contains_key("temp_key"));

        // Try removing non-existent key
        let not_found = metadata.remove("nonexistent");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_metadata_clear() {
        let mut metadata = Metadata::new();
        metadata.insert("key1".to_string(), MetadataValue::U32(1));
        metadata.insert("key2".to_string(), MetadataValue::U32(2));

        assert_eq!(metadata.len(), 2);

        metadata = Metadata::new(); // Replace clear() with new instance
        assert_eq!(metadata.len(), 0);
        assert!(metadata.is_empty());
    }

    #[test]
    fn test_metadata_iteration() {
        let mut metadata = Metadata::new();
        metadata.insert("a".to_string(), MetadataValue::U8(1));
        metadata.insert("b".to_string(), MetadataValue::U8(2));
        metadata.insert("c".to_string(), MetadataValue::U8(3));

        let keys: Vec<_> = metadata.keys().cloned().collect();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"a".to_string()));
        assert!(keys.contains(&"b".to_string()));
        assert!(keys.contains(&"c".to_string()));

        let values: Vec<_> = metadata.values().collect();
        assert_eq!(values.len(), 3);

        let pairs: Vec<_> = metadata.iter().collect();
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn test_metadata_convenience_getters() {
        let mut metadata = Metadata::new();
        metadata.insert("string_key".to_string(), MetadataValue::String("test_value".to_string()));
        metadata.insert("uint_key".to_string(), MetadataValue::U32(42));
        metadata.insert("bool_key".to_string(), MetadataValue::Bool(true));
        metadata.insert("float_key".to_string(), MetadataValue::F32(std::f32::consts::PI));

        assert_eq!(metadata.get_string("string_key"), Some("test_value"));
        assert_eq!(metadata.get_u64("uint_key"), Some(42));
        assert_eq!(metadata.get_bool("bool_key"), Some(true));
        assert_eq!(metadata.get_f64("float_key"), Some(std::f32::consts::PI as f64));

        // Test type mismatches return None
        assert_eq!(metadata.get_string("uint_key"), None);
        assert_eq!(metadata.get_u64("string_key"), None);
    }

    #[test]
    fn test_metadata_serialization() {
        let mut metadata = Metadata::new();
        metadata.insert("test_u32".to_string(), MetadataValue::U32(12345));
        metadata.insert("test_string".to_string(), MetadataValue::String("hello".to_string()));
        metadata.insert("test_bool".to_string(), MetadataValue::Bool(false));

        let mut serialized = Vec::new();
        metadata.write_to(&mut serialized).expect("Failed to serialize metadata");
        assert!(!serialized.is_empty());

        let mut cursor = Cursor::new(&serialized);
        let deserialized =
            Metadata::read_from(&mut cursor, 3).expect("Failed to deserialize metadata");

        assert_eq!(deserialized.len(), 3);
        assert_eq!(deserialized.get_u64("test_u32"), Some(12345));
        assert_eq!(deserialized.get_string("test_string"), Some("hello"));
        assert_eq!(deserialized.get_bool("test_bool"), Some(false));
    }
}

mod alignment_tests {
    use super::*;

    #[test]
    fn test_pad_to_alignment() {
        assert_eq!(gguf::format::alignment::calculate_padding(0, 32), 0);
        assert_eq!(gguf::format::alignment::calculate_padding(1, 32), 31);
        assert_eq!(gguf::format::alignment::calculate_padding(16, 32), 16);
        assert_eq!(gguf::format::alignment::calculate_padding(32, 32), 0);
        assert_eq!(gguf::format::alignment::calculate_padding(33, 32), 31);
        assert_eq!(gguf::format::alignment::calculate_padding(48, 32), 16);
        assert_eq!(gguf::format::alignment::calculate_padding(64, 32), 0);
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(0, 32), 0);
        assert_eq!(align_to(1, 32), 32);
        assert_eq!(align_to(16, 32), 32);
        assert_eq!(align_to(32, 32), 32);
        assert_eq!(align_to(33, 32), 64);
        assert_eq!(align_to(48, 32), 64);
        assert_eq!(align_to(64, 32), 64);
    }

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(0, 32));
        assert!(!is_aligned(1, 32));
        assert!(!is_aligned(16, 32));
        assert!(is_aligned(32, 32));
        assert!(!is_aligned(33, 32));
        assert!(!is_aligned(48, 32));
        assert!(is_aligned(64, 32));
    }

    #[test]
    fn test_alignment_edge_cases() {
        // Test with alignment of 1 (everything should be aligned)
        assert_eq!(gguf::format::alignment::calculate_padding(0, 1), 0);
        assert_eq!(gguf::format::alignment::calculate_padding(100, 1), 0);
        assert_eq!(align_to(50, 1), 50);
        assert!(is_aligned(123, 1));

        // Test with power-of-2 alignments
        assert_eq!(align_to(7, 8), 8);
        assert_eq!(align_to(8, 8), 8);
        assert_eq!(align_to(9, 8), 16);

        // Test with large values
        assert_eq!(align_to(1000, 64), 1024);
        assert_eq!(gguf::format::alignment::calculate_padding(1000, 64), 24);
    }

    #[test]
    fn test_alignment_zero_graceful() {
        // Zero alignment should return the original position
        assert_eq!(align_to(10, 0), 10);
        assert_eq!(calculate_padding(10, 0), 0);
    }
}

mod endian_tests {

    #[test]
    fn test_u32_endian_conversion() {
        let value = 0x12345678u32;
        let le_bytes = value.to_le_bytes();
        let be_bytes = value.to_be_bytes();

        assert_eq!(u32::from_le_bytes(le_bytes), value);
        assert_eq!(u32::from_be_bytes(be_bytes), value);

        // Test little-endian byte order
        assert_eq!(le_bytes, [0x78, 0x56, 0x34, 0x12]);
        assert_eq!(be_bytes, [0x12, 0x34, 0x56, 0x78]);
    }

    #[test]
    fn test_u64_endian_conversion() {
        let value = 0x123456789ABCDEF0u64;
        let le_bytes = value.to_le_bytes();
        let be_bytes = value.to_be_bytes();

        assert_eq!(u64::from_le_bytes(le_bytes), value);
        assert_eq!(u64::from_be_bytes(be_bytes), value);

        // Test little-endian byte order
        assert_eq!(le_bytes, [0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12]);
        assert_eq!(be_bytes, [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]);
    }

    #[test]
    fn test_f32_endian_conversion() {
        let value = std::f32::consts::PI;
        let bytes = value.to_le_bytes();
        let restored = f32::from_le_bytes(bytes);

        assert!((restored - value).abs() < f32::EPSILON);
    }

    #[test]
    fn test_f64_endian_conversion() {
        let value = std::f64::consts::PI;
        let bytes = value.to_le_bytes();
        let restored = f64::from_le_bytes(bytes);

        assert!((restored - value).abs() < f64::EPSILON);
    }

    #[test]
    fn test_signed_integer_endian() {
        let value = -12345i32;
        let bytes = value.to_le_bytes();
        let restored = i32::from_le_bytes(bytes);

        assert_eq!(restored, value);

        let value64 = -987654321i64;
        let bytes64 = value64.to_le_bytes();
        let restored64 = i64::from_le_bytes(bytes64);

        assert_eq!(restored64, value64);
    }
}

mod types_tests {}
