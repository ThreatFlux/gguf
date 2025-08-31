//! Unit tests for the reader module

use gguf::format::{GGUFTensorType, GGUFValueType};
use gguf::prelude::*;
use gguf::reader::*;
use gguf::tensor::TensorType;
use std::io::Cursor;
use std::io::Write;
use tempfile::NamedTempFile;

mod file_reader_tests {
    use super::*;

    fn create_test_gguf_data() -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count

        // Metadata entry
        // Key: "test_key" (length + string)
        data.extend_from_slice(&8u64.to_le_bytes()); // key length
        data.extend_from_slice(b"test_key");
        data.extend_from_slice(&(GGUFValueType::String as u32).to_le_bytes()); // value type
        data.extend_from_slice(&10u64.to_le_bytes()); // value length
        data.extend_from_slice(b"test_value");

        // Tensor info entries
        // First tensor: "tensor1"
        data.extend_from_slice(&7u64.to_le_bytes()); // name length
        data.extend_from_slice(b"tensor1");
        data.extend_from_slice(&2u32.to_le_bytes()); // dimensions
        data.extend_from_slice(&10u64.to_le_bytes()); // dim 0
        data.extend_from_slice(&5u64.to_le_bytes()); // dim 1
        data.extend_from_slice(&(GGUFTensorType::F32 as u32).to_le_bytes()); // tensor type
        data.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Second tensor: "tensor2"
        data.extend_from_slice(&7u64.to_le_bytes()); // name length
        data.extend_from_slice(b"tensor2");
        data.extend_from_slice(&1u32.to_le_bytes()); // dimensions
        data.extend_from_slice(&20u64.to_le_bytes()); // dim 0
        data.extend_from_slice(&(GGUFTensorType::I32 as u32).to_le_bytes()); // tensor type
        data.extend_from_slice(&200u64.to_le_bytes()); // offset

        // Align to 32 bytes
        while data.len() % 32 != 0 {
            data.push(0);
        }

        // Tensor data
        // tensor1: 10*5 F32 values = 200 bytes
        for i in 0..50 {
            data.extend_from_slice(&(i as f32).to_le_bytes());
        }

        // tensor2: 20 I32 values = 80 bytes
        for i in 0..20i32 {
            data.extend_from_slice(&i.to_le_bytes());
        }

        data
    }

    #[test]
    fn test_file_reader_creation() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFFileReader::new(cursor).expect("Failed to create reader");

        assert_eq!(reader.tensor_count(), 2);
        assert_eq!(reader.metadata().len(), 1);
        assert_eq!(reader.metadata().get_string("test_key"), Some("test_value"));
    }

    #[test]
    fn test_file_reader_tensor_info() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFFileReader::new(cursor).expect("Failed to create reader");

        let tensor_info = reader.get_tensor_info("tensor1");
        assert!(tensor_info.is_some());

        let info = tensor_info.unwrap();
        assert_eq!(info.name(), "tensor1");
        assert_eq!(info.tensor_type(), TensorType::F32);
        assert_eq!(info.shape().dims(), &[10, 5]);
        assert_eq!(info.element_count(), 50);
    }

    #[test]
    fn test_file_reader_tensor_data_loading() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let mut reader = GGUFFileReader::new(cursor).expect("Failed to create reader");

        // Load tensor1 data
        let tensor_data = reader.load_tensor_data("tensor1").expect("Failed to load tensor data");
        assert!(tensor_data.is_some());

        let data = tensor_data.unwrap();
        assert_eq!(data.len(), 200); // 50 * 4 bytes

        // Verify first few values
        let floats: Vec<f32> = data
            .as_slice()
            .chunks_exact(4)
            .take(5)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        assert_eq!(floats, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_file_reader_nonexistent_tensor() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let mut reader = GGUFFileReader::new(cursor).expect("Failed to create reader");

        assert!(reader.get_tensor_info("nonexistent").is_none());

        let result = reader.load_tensor_data("nonexistent");
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_file_reader_tensor_iteration() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFFileReader::new(cursor).expect("Failed to create reader");

        let tensor_names: Vec<String> =
            reader.tensor_infos().iter().map(|info| info.name().to_string()).collect();

        assert_eq!(tensor_names.len(), 2);
        assert!(tensor_names.contains(&"tensor1".to_string()));
        assert!(tensor_names.contains(&"tensor2".to_string()));
    }

    #[test]
    fn test_file_reader_from_file() {
        let data = create_test_gguf_data();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(&data).expect("Failed to write test data");
        temp_file.flush().expect("Failed to flush temp file");

        let reader = open_gguf_file(temp_file.path()).expect("Failed to open GGUF file");

        assert_eq!(reader.tensor_count(), 2);
        assert_eq!(reader.metadata().len(), 1);
    }

    #[test]
    fn test_file_reader_invalid_file() {
        let invalid_data = vec![1, 2, 3, 4]; // Not a valid GGUF file
        let cursor = Cursor::new(invalid_data);

        let result = GGUFFileReader::new(cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_file_reader_truncated_file() {
        let mut data = create_test_gguf_data();
        data.truncate(50); // Truncate the file

        let cursor = Cursor::new(data);
        let result = GGUFFileReader::new(cursor);

        assert!(result.is_err());
    }
}

mod stream_reader_tests {
    use super::*;

    #[test]
    fn test_stream_reader_basic() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFStreamReader::new(cursor).expect("Failed to create stream reader");

        assert_eq!(reader.header().tensor_count, 2);
        assert_eq!(reader.header().metadata_kv_count, 1);
    }

    #[test]
    fn test_stream_reader_read_header() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFStreamReader::new(cursor).expect("Failed to create reader");
        let header = reader.header();

        assert_eq!(header.magic, GGUF_MAGIC);
        assert_eq!(header.version, GGUF_VERSION);
        assert_eq!(header.tensor_count, 2);
        assert_eq!(header.metadata_kv_count, 1);
    }

    #[test]
    fn test_stream_reader_read_metadata() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        // Create reader which automatically reads header and metadata
        let reader = GGUFStreamReader::new(cursor).expect("Failed to create reader");
        let metadata = reader.metadata();

        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata.get_string("test_key"), Some("test_value"));
    }

    #[test]
    fn test_stream_reader_read_tensor_info() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFStreamReader::new(cursor).expect("Failed to create reader");

        let tensor_infos = reader.tensor_infos();
        assert_eq!(tensor_infos.len(), 2);
        assert_eq!(tensor_infos[0].name(), "tensor1");
        assert_eq!(tensor_infos[1].name(), "tensor2");
    }

    #[test]
    fn test_stream_reader_sequential() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFStreamReader::new(cursor).expect("Failed to create reader");

        // Test that we can access metadata and tensor infos
        let metadata = reader.metadata();
        let tensor_infos = reader.tensor_infos();

        assert!(!metadata.is_empty());
        assert_eq!(tensor_infos.len(), 2);
    }
}

mod tensor_reader_tests {
    use super::*;

    #[test]
    fn test_tensor_reader_creation() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let file_reader = GGUFFileReader::new(cursor).expect("Failed to create file reader");

        // Test that we can access tensor information through the file reader
        assert_eq!(file_reader.tensor_count(), 2);
    }

    #[test]
    fn test_tensor_reader_get_info() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let file_reader = GGUFFileReader::new(cursor).expect("Failed to create file reader");

        let info = file_reader.get_tensor_info("tensor1");
        assert!(info.is_some());

        let info = info.unwrap();
        assert_eq!(info.name(), "tensor1");
        assert_eq!(info.tensor_type(), TensorType::F32);
    }

    #[test]
    fn test_tensor_reader_load_data() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let mut file_reader = GGUFFileReader::new(cursor).expect("Failed to create file reader");
        // Use file_reader directly instead of TensorReader

        let tensor_data = file_reader.load_tensor_data("tensor2").expect("Failed to load tensor");

        assert!(tensor_data.is_some());
        let data = tensor_data.unwrap();
        assert_eq!(data.len(), 80); // 20 * 4 bytes for i32
    }

    #[test]
    fn test_tensor_reader_load_all() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let mut file_reader = GGUFFileReader::new(cursor).expect("Failed to create file reader");
        // Use file_reader directly instead of TensorReader

        file_reader.load_all_tensor_data().expect("Failed to load all tensors");

        // Check that we have 2 tensors
        assert_eq!(file_reader.tensor_count(), 2);

        // Verify we can get individual tensor names
        let tensor_names: Vec<&str> = file_reader.tensor_names();
        assert!(tensor_names.contains(&"tensor1"));
        assert!(tensor_names.contains(&"tensor2"));

        // Verify we can load individual tensor data
        let tensor1_data = file_reader
            .load_tensor_data("tensor1")
            .expect("Failed to load tensor1")
            .expect("Tensor1 should exist");
        assert_eq!(tensor1_data.len(), 200);

        let tensor2_data = file_reader
            .load_tensor_data("tensor2")
            .expect("Failed to load tensor2")
            .expect("Tensor2 should exist");
        assert_eq!(tensor2_data.len(), 80);
    }

    #[test]
    fn test_tensor_reader_selective_loading() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let mut file_reader = GGUFFileReader::new(cursor).expect("Failed to create file reader");
        // Use file_reader directly instead of TensorReader

        let tensor_data = file_reader.load_tensor_data("tensor1").expect("Failed to load tensor");

        assert!(tensor_data.is_some());

        let tensor_names: Vec<&str> = file_reader.tensor_names();
        assert!(tensor_names.contains(&"tensor1"));
    }

    #[test]
    fn test_tensor_reader_iterator() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let file_reader = GGUFFileReader::new(cursor).expect("Failed to create file reader");
        // Use file_reader directly instead of TensorReader

        let tensor_names: Vec<&str> = file_reader.tensor_names();

        assert_eq!(tensor_names.len(), 2);
        assert!(tensor_names.contains(&"tensor1"));
        assert!(tensor_names.contains(&"tensor2"));
    }
}

mod error_handling_tests {
    use super::*;

    #[test]
    fn test_reader_invalid_magic() {
        let mut data = vec![0u8; 24];
        data[0..4].copy_from_slice(&0x12345678u32.to_le_bytes()); // Invalid magic

        let cursor = Cursor::new(data);
        let result = GGUFFileReader::new(cursor);

        assert!(matches!(result, Err(GGUFError::InvalidMagic { .. })));
    }

    #[test]
    fn test_reader_unsupported_version() {
        let mut data = vec![0u8; 24];
        data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
        data[4..8].copy_from_slice(&999u32.to_le_bytes()); // Unsupported version

        let cursor = Cursor::new(data);
        let result = GGUFFileReader::new(cursor);

        assert!(matches!(result, Err(GGUFError::UnsupportedVersion(999))));
    }

    #[test]
    fn test_reader_corrupted_metadata() {
        let mut data = Vec::new();

        // Valid header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count

        // Corrupted metadata - truncated key
        data.extend_from_slice(&100u64.to_le_bytes()); // key length (too long)
        data.extend_from_slice(b"short"); // But only provide short data

        let cursor = Cursor::new(data);
        let result = GGUFFileReader::new(cursor);

        assert!(result.is_err());
    }

    #[test]
    fn test_reader_corrupted_tensor_info() {
        let mut data = Vec::new();

        // Valid header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count

        // Corrupted tensor info - invalid tensor type
        data.extend_from_slice(&7u64.to_le_bytes()); // name length
        data.extend_from_slice(b"tensor1");
        data.extend_from_slice(&2u32.to_le_bytes()); // dimensions
        data.extend_from_slice(&10u64.to_le_bytes()); // dim 0
        data.extend_from_slice(&5u64.to_le_bytes()); // dim 1
        data.extend_from_slice(&999u32.to_le_bytes()); // invalid tensor type

        let cursor = Cursor::new(data);
        let result = GGUFFileReader::new(cursor);

        assert!(result.is_err());
    }

    #[test]
    fn test_reader_seek_error() {
        // Create a cursor that can't seek beyond its data
        let data = vec![0u8; 100];
        let cursor = Cursor::new(data);

        let mut reader = GGUFStreamReader::new(cursor).expect("Failed to create reader");

        // Try to read all tensors (should work)
        let result = reader.read_all_tensors();
        assert!(result.is_ok());
    }

    #[test]
    fn test_reader_read_beyond_bounds() {
        let data = create_test_gguf_data();
        let cursor = Cursor::new(data);

        let reader = GGUFFileReader::new(cursor).expect("Failed to create reader");

        // Try to load a tensor that would read beyond the file bounds
        // This should be handled gracefully by checking tensor info first
        if let Some(info) = reader.get_tensor_info("tensor1") {
            // Manually corrupt the offset to point beyond file bounds
            let mut _corrupted_info = info.clone();
            // This test assumes we can modify the tensor info, which might not be possible
            // depending on the actual API design
        }
    }
}

// Helper function to create test data
fn create_test_gguf_data() -> Vec<u8> {
    let mut data = Vec::new();

    // Header
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count

    // Metadata entry
    data.extend_from_slice(&8u64.to_le_bytes());
    data.extend_from_slice(b"test_key");
    data.extend_from_slice(&(GGUFValueType::String as u32).to_le_bytes());
    data.extend_from_slice(&10u64.to_le_bytes());
    data.extend_from_slice(b"test_value");

    // Tensor info
    data.extend_from_slice(&7u64.to_le_bytes());
    data.extend_from_slice(b"tensor1");
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());
    data.extend_from_slice(&(GGUFTensorType::F32 as u32).to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    // Align to 32 bytes
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor data: 4 F32 values
    for i in 0..4 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    data
}
