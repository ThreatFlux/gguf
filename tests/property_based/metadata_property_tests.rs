//! Property-based tests for metadata operations

use gguf::builder::GGUFBuilder;
use gguf::format::metadata::MetadataValue;
use proptest::prelude::*;
use std::io::Cursor;

// Strategy for generating metadata keys
fn metadata_key_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z][a-zA-Z0-9._-]{0,30}" // Valid metadata key pattern
}

// Strategy for generating string values
fn string_value_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9 ._-]{0,100}" // Reasonable string content
}

proptest! {
    #[test]
    fn test_metadata_string_round_trip(
        key in metadata_key_strategy(),
        value in string_value_strategy()
    ) {
        let builder = GGUFBuilder::new()
            .add_metadata(&key, MetadataValue::String(value.clone()));

        let (bytes, _) = builder.build_to_bytes().expect("Failed to build");

        let cursor = Cursor::new(bytes);
        let reader = gguf::reader::GGUFFileReader::new(cursor).expect("Failed to read");

        prop_assert_eq!(reader.metadata().get_string(&key), Some(value.as_str()));
    }

    #[test]
    fn test_metadata_numeric_round_trip(
        key in metadata_key_strategy(),
        value in any::<u32>()
    ) {
        let builder = GGUFBuilder::new()
            .add_metadata(&key, MetadataValue::U32(value));

        let (bytes, _) = builder.build_to_bytes().expect("Failed to build");

        let cursor = Cursor::new(bytes);
        let reader = gguf::reader::GGUFFileReader::new(cursor).expect("Failed to read");

        prop_assert_eq!(reader.metadata().get_u64(&key), Some(value as u64));
    }

    #[test]
    fn test_metadata_bool_round_trip(
        key in metadata_key_strategy(),
        value in any::<bool>()
    ) {
        let builder = GGUFBuilder::new()
            .add_metadata(&key, MetadataValue::Bool(value));

        let (bytes, _) = builder.build_to_bytes().expect("Failed to build");

        let cursor = Cursor::new(bytes);
        let reader = gguf::reader::GGUFFileReader::new(cursor).expect("Failed to read");

        prop_assert_eq!(reader.metadata().get_bool(&key), Some(value));
    }

    #[test]
    fn test_metadata_f32_round_trip(
        key in metadata_key_strategy(),
        value in -1000.0f32..1000.0f32
    ) {
        let builder = GGUFBuilder::new()
            .add_metadata(&key, MetadataValue::F32(value));

        let (bytes, _) = builder.build_to_bytes().expect("Failed to build");

        let cursor = Cursor::new(bytes);
        let reader = gguf::reader::GGUFFileReader::new(cursor).expect("Failed to read");

        let loaded_value = reader.metadata().get_f64(&key).unwrap() as f32;
        prop_assert!((loaded_value - value).abs() < f32::EPSILON);
    }

    #[test]
    fn test_multiple_metadata_entries(
        entries in prop::collection::vec(
            (metadata_key_strategy(), string_value_strategy()),
            1..20
        )
    ) {
        // Ensure unique keys
        let mut unique_entries = std::collections::HashMap::new();
        for (key, value) in entries {
            unique_entries.insert(key, value);
        }
        prop_assume!(!unique_entries.is_empty());

        let mut builder = GGUFBuilder::new();
        for (key, value) in &unique_entries {
            builder = builder.add_metadata(key, MetadataValue::String(value.clone()));
        }

        let (bytes, _) = builder.build_to_bytes().expect("Failed to build");

        let cursor = Cursor::new(bytes);
        let reader = gguf::reader::GGUFFileReader::new(cursor).expect("Failed to read");

        prop_assert_eq!(reader.metadata().len(), unique_entries.len());

        for (key, expected_value) in &unique_entries {
            prop_assert_eq!(reader.metadata().get_string(key), Some(expected_value.as_str()));
        }
    }

    #[test]
    fn test_metadata_type_mixing(
        string_key in metadata_key_strategy(),
        string_value in string_value_strategy(),
        int_key in metadata_key_strategy().prop_filter("Different from string_key", |k| k != "string_key"),
        int_value in any::<u32>(),
        bool_key in metadata_key_strategy().prop_filter("Different from others", |k| k != "string_key" && k != "int_key"),
        bool_value in any::<bool>()
    ) {
        let builder = GGUFBuilder::new()
            .add_metadata(&string_key, MetadataValue::String(string_value.clone()))
            .add_metadata(&int_key, MetadataValue::U64(int_value as u64))
            .add_metadata(&bool_key, MetadataValue::Bool(bool_value));

        let (bytes, _) = builder.build_to_bytes().expect("Failed to build");

        let cursor = Cursor::new(bytes);
        let reader = gguf::reader::GGUFFileReader::new(cursor).expect("Failed to read");

        prop_assert_eq!(reader.metadata().len(), 3);
        prop_assert_eq!(reader.metadata().get_string(&string_key), Some(string_value.as_str()));
        prop_assert_eq!(reader.metadata().get_u64(&int_key), Some(int_value as u64));
        prop_assert_eq!(reader.metadata().get_bool(&bool_key), Some(bool_value));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))] // Fewer cases for complex tests

    #[test]
    fn test_metadata_size_limits(
        key in metadata_key_strategy(),
        size in 0usize..10000
    ) {
        let large_string = "x".repeat(size);

        let builder = GGUFBuilder::new()
            .add_metadata(&key, MetadataValue::String(large_string.clone()));

        let result = builder.build_to_bytes();

        if size > 0 {
            prop_assert!(result.is_ok());

            let (bytes, _) = result.unwrap();
            let cursor = Cursor::new(bytes);
            let reader = gguf::reader::GGUFFileReader::new(cursor).expect("Failed to read");

            prop_assert_eq!(reader.metadata().get_string(&key), Some(large_string.as_str()));
        }
    }

    #[test]
    fn test_metadata_key_variations(
        base_key in "[a-z]{1,10}",
        suffix in prop::option::of("[._-][a-z0-9]{1,5}")
    ) {
        let key = match suffix {
            Some(s) => format!("{}{}", base_key, s),
            None => base_key,
        };

        let builder = GGUFBuilder::new()
            .add_metadata(&key, MetadataValue::U32(42));

        let (bytes, _) = builder.build_to_bytes().expect("Failed to build");

        let cursor = Cursor::new(bytes);
        let reader = gguf::reader::GGUFFileReader::new(cursor).expect("Failed to read");

        prop_assert_eq!(reader.metadata().get_u64(&key), Some(42));
    }
}
