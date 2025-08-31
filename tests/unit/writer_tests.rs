//! Unit tests for the writer module

use gguf::writer::*;
use gguf::prelude::*;
use gguf::reader::GGUFFileReader;
use std::io::{Cursor, Seek, SeekFrom};
use tempfile::NamedTempFile;
use std::io::Write as IoWrite;

mod file_writer_tests {
    use super::*;

    #[test]
    fn test_file_writer_creation() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let writer = GGUFFileWriter::new(cursor).expect("Failed to create writer");
        
        // Writer should be created successfully
        assert_eq!(writer.bytes_written(), 0);
    }

    #[test]
    fn test_file_writer_write_header() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFFileWriter::new(cursor).expect("Failed to create writer");
        
        let header = GGUFHeader::new(2, 1);
        writer.write_header(&header).expect("Failed to write header");
        
        assert_eq!(writer.bytes_written(), 24); // Header is 24 bytes
        assert_eq!(buffer.len(), 24);
        
        // Verify header content
        assert_eq!(&buffer[0..4], &GGUF_MAGIC.to_le_bytes());
        assert_eq!(&buffer[4..8], &GGUF_VERSION.to_le_bytes());
        
        let tensor_count = u64::from_le_bytes([
            buffer[8], buffer[9], buffer[10], buffer[11],
            buffer[12], buffer[13], buffer[14], buffer[15]
        ]);
        assert_eq!(tensor_count, 2);
        
        let metadata_count = u64::from_le_bytes([
            buffer[16], buffer[17], buffer[18], buffer[19],
            buffer[20], buffer[21], buffer[22], buffer[23]
        ]);
        assert_eq!(metadata_count, 1);
    }

    #[test]
    fn test_file_writer_write_metadata() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFFileWriter::new(cursor).expect("Failed to create writer");
        
        let mut metadata = Metadata::new();
        metadata.insert("test_key".to_string(), MetadataValue::String("test_value".to_string()));
        metadata.insert("num_key".to_string(), MetadataValue::UInt32(42));
        
        writer.write_metadata(&metadata).expect("Failed to write metadata");
        
        assert!(writer.bytes_written() > 0);
        assert!(!buffer.is_empty());
        
        // Should be able to read back the metadata
        let mut cursor = Cursor::new(&buffer);
        let read_metadata = Metadata::read(&mut cursor, 2).expect("Failed to read metadata");
        
        assert_eq!(read_metadata.len(), 2);
        assert_eq!(read_metadata.get_string("test_key"), Some("test_value"));
        assert_eq!(read_metadata.get_u32("num_key"), Some(42));
    }

    #[test]
    fn test_file_writer_write_tensor_info() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFFileWriter::new(cursor).expect("Failed to create writer");
        
        let tensor_info = TensorInfo::new(
            "test_tensor".to_string(),
            TensorType::F32,
            TensorShape::new(vec![10, 5]),
            1024,
        );
        
        writer.write_tensor_info(&tensor_info).expect("Failed to write tensor info");
        
        assert!(writer.bytes_written() > 0);
        assert!(!buffer.is_empty());
        
        // Verify the data was written correctly
        let mut cursor = Cursor::new(&buffer);
        let read_info = TensorInfo::read(&mut cursor).expect("Failed to read tensor info");
        
        assert_eq!(read_info.name(), tensor_info.name());
        assert_eq!(read_info.tensor_type(), tensor_info.tensor_type());
        assert_eq!(read_info.shape(), tensor_info.shape());
        assert_eq!(read_info.offset(), tensor_info.offset());
    }

    #[test]
    fn test_file_writer_write_tensor_data() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFFileWriter::new(cursor).expect("Failed to create writer");
        
        let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let tensor_data = TensorData::F32(data.clone());
        
        writer.write_tensor_data(&tensor_data).expect("Failed to write tensor data");
        
        assert_eq!(writer.bytes_written(), 16); // 4 * 4 bytes
        assert_eq!(buffer.len(), 16);
        
        // Verify the data was written correctly
        for (i, chunk) in buffer.chunks_exact(4).enumerate() {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            assert_eq!(value, data[i]);
        }
    }

    #[test]
    fn test_file_writer_alignment() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFFileWriter::new(cursor).expect("Failed to create writer");
        
        // Write some data that's not aligned
        let data = vec![1u8, 2u8, 3u8]; // 3 bytes
        let tensor_data = TensorData::Bytes(data);
        writer.write_tensor_data(&tensor_data).expect("Failed to write data");
        
        assert_eq!(writer.bytes_written(), 3);
        
        // Write alignment padding
        writer.write_alignment_padding(DEFAULT_ALIGNMENT).expect("Failed to write padding");
        
        // Should now be aligned to 32 bytes
        assert_eq!(writer.bytes_written(), 32);
        assert_eq!(buffer.len(), 32);
        
        // Check that padding bytes are zeros
        for &byte in &buffer[3..32] {
            assert_eq!(byte, 0);
        }
    }

    #[test]
    fn test_file_writer_complete_file() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFFileWriter::new(cursor).expect("Failed to create writer");
        
        // Write a complete GGUF file
        let header = GGUFHeader::new(1, 1);
        writer.write_header(&header).expect("Failed to write header");
        
        let mut metadata = Metadata::new();
        metadata.insert("model_name".to_string(), MetadataValue::String("test_model".to_string()));
        writer.write_metadata(&metadata).expect("Failed to write metadata");
        
        let tensor_info = TensorInfo::new(
            "weights".to_string(),
            TensorType::F32,
            TensorShape::new(vec![2, 2]),
            0, // Will be updated later
        );
        writer.write_tensor_info(&tensor_info).expect("Failed to write tensor info");
        
        // Align before tensor data
        writer.write_alignment_padding(DEFAULT_ALIGNMENT).expect("Failed to write padding");
        
        let tensor_data = TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]);
        writer.write_tensor_data(&tensor_data).expect("Failed to write tensor data");
        
        writer.finish().expect("Failed to finish writing");
        
        // Verify we can read back the file
        let cursor = Cursor::new(&buffer);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read back the file");
        
        assert_eq!(reader.tensor_count(), 1);
        assert_eq!(reader.metadata().len(), 1);
        assert_eq!(reader.metadata().get_string("model_name"), Some("test_model"));
        
        let tensor_info = reader.get_tensor_info("weights").unwrap();
        assert_eq!(tensor_info.tensor_type(), TensorType::F32);
        assert_eq!(tensor_info.shape().dimensions(), &[2, 2]);
    }

    #[test]
    fn test_file_writer_to_file() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let file_path = temp_file.path();
        
        // Create a GGUF file
        let result = create_gguf_file(file_path).expect("Failed to create file");
        let mut writer = result.writer;
        
        let header = GGUFHeader::new(1, 0);
        writer.write_header(&header).expect("Failed to write header");
        
        let tensor_info = TensorInfo::new(
            "test".to_string(),
            TensorType::I32,
            TensorShape::new(vec![3]),
            0,
        );
        writer.write_tensor_info(&tensor_info).expect("Failed to write tensor info");
        
        writer.write_alignment_padding(DEFAULT_ALIGNMENT).expect("Failed to write padding");
        
        let tensor_data = TensorData::I32(vec![10, 20, 30]);
        writer.write_tensor_data(&tensor_data).expect("Failed to write tensor data");
        
        writer.finish().expect("Failed to finish");
        
        // Verify the file was written correctly
        use gguf::reader::open_gguf_file;
        let reader = open_gguf_file(file_path).expect("Failed to read file");
        assert_eq!(reader.tensor_count(), 1);
    }

    #[test]
    fn test_file_writer_error_handling() {
        // Test writing to a read-only cursor
        let buffer = Vec::new();
        let cursor = Cursor::new(buffer);
        
        let mut writer = GGUFFileWriter::new(cursor).expect("Failed to create writer");
        
        // This should work fine since Cursor<Vec<u8>> is writable
        let header = GGUFHeader::default();
        writer.write_header(&header).expect("Should succeed");
        
        // Test with invalid metadata
        let mut invalid_metadata = Metadata::new();
        // Add metadata that might cause serialization issues
        invalid_metadata.insert("".to_string(), MetadataValue::String("".to_string()));
        
        // Should still work with empty strings
        writer.write_metadata(&invalid_metadata).expect("Should handle empty strings");
    }
}

mod stream_writer_tests {
    use super::*;

    #[test]
    fn test_stream_writer_creation() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let writer = GGUFStreamWriter::new(cursor);
        
        assert_eq!(writer.position(), 0);
    }

    #[test]
    fn test_stream_writer_write_bytes() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFStreamWriter::new(cursor);
        
        let data = vec![1, 2, 3, 4, 5];
        writer.write_bytes(&data).expect("Failed to write bytes");
        
        assert_eq!(writer.position(), 5);
        assert_eq!(buffer, data);
    }

    #[test]
    fn test_stream_writer_write_u32() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFStreamWriter::new(cursor);
        
        writer.write_u32(0x12345678).expect("Failed to write u32");
        
        assert_eq!(writer.position(), 4);
        assert_eq!(buffer, &0x12345678u32.to_le_bytes());
    }

    #[test]
    fn test_stream_writer_write_u64() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFStreamWriter::new(cursor);
        
        writer.write_u64(0x123456789ABCDEF0).expect("Failed to write u64");
        
        assert_eq!(writer.position(), 8);
        assert_eq!(buffer, &0x123456789ABCDEF0u64.to_le_bytes());
    }

    #[test]
    fn test_stream_writer_write_f32() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFStreamWriter::new(cursor);
        
        let value = 3.14159f32;
        writer.write_f32(value).expect("Failed to write f32");
        
        assert_eq!(writer.position(), 4);
        assert_eq!(buffer, &value.to_le_bytes());
    }

    #[test]
    fn test_stream_writer_write_string() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFStreamWriter::new(cursor);
        
        let text = "Hello, World!";
        writer.write_string(text).expect("Failed to write string");
        
        assert_eq!(writer.position(), 8 + text.len() as u64); // length + data
        
        // Verify format: length (8 bytes) + string data
        let expected_len = text.len() as u64;
        assert_eq!(&buffer[0..8], &expected_len.to_le_bytes());
        assert_eq!(&buffer[8..], text.as_bytes());
    }

    #[test]
    fn test_stream_writer_seek() {
        let mut buffer = vec![0u8; 100];
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFStreamWriter::new(cursor);
        
        // Write some data
        writer.write_u32(0x12345678).expect("Failed to write");
        assert_eq!(writer.position(), 4);
        
        // Seek to a different position
        writer.seek_to(10).expect("Failed to seek");
        assert_eq!(writer.position(), 10);
        
        // Write more data
        writer.write_u32(0x87654321).expect("Failed to write");
        assert_eq!(writer.position(), 14);
        
        // Verify the data was written at the correct positions
        assert_eq!(&buffer[0..4], &0x12345678u32.to_le_bytes());
        assert_eq!(&buffer[10..14], &0x87654321u32.to_le_bytes());
    }

    #[test]
    fn test_stream_writer_alignment() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFStreamWriter::new(cursor);
        
        // Write 3 bytes
        writer.write_bytes(&[1, 2, 3]).expect("Failed to write");
        assert_eq!(writer.position(), 3);
        
        // Align to 8 bytes
        writer.align_to(8).expect("Failed to align");
        assert_eq!(writer.position(), 8);
        
        // Check padding
        assert_eq!(buffer, &[1, 2, 3, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_stream_writer_flush() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GGUFStreamWriter::new(cursor);
        
        writer.write_bytes(&[1, 2, 3, 4]).expect("Failed to write");
        writer.flush().expect("Failed to flush");
        
        // Data should be available immediately
        assert_eq!(buffer, &[1, 2, 3, 4]);
    }
}

mod tensor_writer_tests {
    use super::*;

    #[test]
    fn test_tensor_writer_creation() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let file_writer = GGUFFileWriter::new(cursor).expect("Failed to create file writer");
        let tensor_writer = GGUFTensorWriter::new(file_writer);
        
        assert_eq!(tensor_writer.tensor_count(), 0);
    }

    #[test]
    fn test_tensor_writer_add_tensor() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let file_writer = GGUFFileWriter::new(cursor).expect("Failed to create file writer");
        let mut tensor_writer = GGUFTensorWriter::new(file_writer);
        
        let data = TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]);
        let info = TensorInfo::new(
            "test_tensor".to_string(),
            TensorType::F32,
            TensorShape::new(vec![2, 2]),
            0,
        );
        
        tensor_writer.add_tensor(info, data).expect("Failed to add tensor");
        
        assert_eq!(tensor_writer.tensor_count(), 1);
    }

    #[test]
    fn test_tensor_writer_write_all() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let file_writer = GGUFFileWriter::new(cursor).expect("Failed to create file writer");
        let mut tensor_writer = GGUFTensorWriter::new(file_writer);
        
        // Add multiple tensors
        let data1 = TensorData::F32(vec![1.0, 2.0]);
        let info1 = TensorInfo::new(
            "tensor1".to_string(),
            TensorType::F32,
            TensorShape::new(vec![2]),
            0,
        );
        
        let data2 = TensorData::I32(vec![10, 20, 30]);
        let info2 = TensorInfo::new(
            "tensor2".to_string(),
            TensorType::I32,
            TensorShape::new(vec![3]),
            0,
        );
        
        tensor_writer.add_tensor(info1, data1).expect("Failed to add tensor1");
        tensor_writer.add_tensor(info2, data2).expect("Failed to add tensor2");
        
        // Write metadata
        let mut metadata = Metadata::new();
        metadata.insert("name".to_string(), MetadataValue::String("test".to_string()));
        
        let result = tensor_writer.write_to_file(metadata).expect("Failed to write file");
        
        assert!(result.total_bytes_written > 0);
        assert!(result.tensor_data_offset > 0);
        
        // Verify we can read back the file
        let cursor = Cursor::new(&buffer);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read file");
        
        assert_eq!(reader.tensor_count(), 2);
        assert_eq!(reader.metadata().len(), 1);
        assert_eq!(reader.metadata().get_string("name"), Some("test"));
        
        assert!(reader.get_tensor_info("tensor1").is_some());
        assert!(reader.get_tensor_info("tensor2").is_some());
    }

    #[test]
    fn test_tensor_writer_quantized_data() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let file_writer = GGUFFileWriter::new(cursor).expect("Failed to create file writer");
        let mut tensor_writer = GGUFTensorWriter::new(file_writer);
        
        // Create quantized tensor data (simulated)
        let quantized_data = vec![0u8; 64]; // 64 bytes of quantized data
        let data = TensorData::Bytes(quantized_data);
        
        let info = TensorInfo::new(
            "quantized_tensor".to_string(),
            TensorType::Q4_0,
            TensorShape::new(vec![128]), // 128 elements, 4 blocks
            0,
        );
        
        tensor_writer.add_tensor(info, data).expect("Failed to add quantized tensor");
        
        let metadata = Metadata::new();
        let result = tensor_writer.write_to_file(metadata).expect("Failed to write file");
        
        assert!(result.total_bytes_written > 0);
        
        // Verify the file can be read back
        let cursor = Cursor::new(&buffer);
        let reader = GGUFFileReader::new(cursor).expect("Failed to read file");
        
        let tensor_info = reader.get_tensor_info("quantized_tensor").unwrap();
        assert_eq!(tensor_info.tensor_type(), TensorType::Q4_0);
        assert_eq!(tensor_info.shape().dimensions(), &[128]);
    }

    #[test]
    fn test_tensor_writer_large_tensor() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let file_writer = GGUFFileWriter::new(cursor).expect("Failed to create file writer");
        let mut tensor_writer = GGUFTensorWriter::new(file_writer);
        
        // Create a larger tensor
        let large_data: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        let data = TensorData::F32(large_data.clone());
        
        let info = TensorInfo::new(
            "large_tensor".to_string(),
            TensorType::F32,
            TensorShape::new(vec![100, 100]),
            0,
        );
        
        tensor_writer.add_tensor(info, data).expect("Failed to add large tensor");
        
        let metadata = Metadata::new();
        let result = tensor_writer.write_to_file(metadata).expect("Failed to write file");
        
        assert_eq!(result.tensor_count, 1);
        assert!(result.total_bytes_written > 40000); // At least 10000 * 4 bytes for data
        
        // Verify correctness
        let cursor = Cursor::new(&buffer);
        let mut reader = GGUFFileReader::new(cursor).expect("Failed to read file");
        
        let loaded_data = reader.load_tensor_data("large_tensor").expect("Failed to load data").unwrap();
        assert_eq!(loaded_data.len(), 40000); // 10000 * 4 bytes
        
        // Verify first few values
        let floats: Vec<f32> = loaded_data.as_slice()
            .chunks_exact(4)
            .take(5)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        assert_eq!(floats, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_writer_empty_tensor() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let file_writer = GGUFFileWriter::new(cursor).expect("Failed to create file writer");
        let mut tensor_writer = GGUFTensorWriter::new(file_writer);
        
        // Create empty tensor
        let data = TensorData::F32(vec![]);
        let info = TensorInfo::new(
            "empty_tensor".to_string(),
            TensorType::F32,
            TensorShape::new(vec![0]),
            0,
        );
        
        tensor_writer.add_tensor(info, data).expect("Failed to add empty tensor");
        
        let metadata = Metadata::new();
        let result = tensor_writer.write_to_file(metadata).expect("Failed to write file");
        
        assert_eq!(result.tensor_count, 1);
        
        // Verify the file can be read
        let cursor = Cursor::new(&buffer);
        let mut reader = GGUFFileReader::new(cursor).expect("Failed to read file");
        
        let tensor_info = reader.get_tensor_info("empty_tensor").unwrap();
        assert_eq!(tensor_info.element_count(), 0);
        
        let loaded_data = reader.load_tensor_data("empty_tensor").expect("Failed to load data");
        assert!(loaded_data.is_some());
        assert_eq!(loaded_data.unwrap().len(), 0);
    }
}

mod write_result_tests {
    use super::*;

    #[test]
    fn test_write_result_creation() {
        let result = WriteResult {
            total_bytes_written: 1024,
            header_bytes_written: 24,
            metadata_bytes_written: 100,
            tensor_info_bytes_written: 200,
            tensor_data_bytes_written: 700,
            tensor_data_offset: 324,
            tensor_count: 5,
            metadata_count: 3,
        };
        
        assert_eq!(result.total_bytes_written, 1024);
        assert_eq!(result.tensor_count, 5);
        assert_eq!(result.metadata_count, 3);
        
        // Verify the sum
        let expected_total = result.header_bytes_written + 
                           result.metadata_bytes_written + 
                           result.tensor_info_bytes_written + 
                           result.tensor_data_bytes_written;
        assert_eq!(result.total_bytes_written, expected_total);
    }

    #[test]
    fn test_write_result_display() {
        let result = WriteResult {
            total_bytes_written: 1000,
            header_bytes_written: 24,
            metadata_bytes_written: 100,
            tensor_info_bytes_written: 200,
            tensor_data_bytes_written: 676,
            tensor_data_offset: 324,
            tensor_count: 2,
            metadata_count: 1,
        };
        
        let display_str = format!("{}", result);
        assert!(display_str.contains("1000"));
        assert!(display_str.contains("2"));
        assert!(display_str.contains("1"));
    }
}

#[cfg(test)]
mod integration_write_read_tests {
    use super::*;

    #[test]
    fn test_write_then_read_cycle() {
        let mut buffer = Vec::new();
        
        // Write phase
        {
            let cursor = Cursor::new(&mut buffer);
            let file_writer = GGUFFileWriter::new(cursor).expect("Failed to create writer");
            let mut tensor_writer = GGUFTensorWriter::new(file_writer);
            
            // Add test data
            let data = TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]);
            let info = TensorInfo::new(
                "test".to_string(),
                TensorType::F32,
                TensorShape::new(vec![2, 2]),
                0,
            );
            tensor_writer.add_tensor(info, data).expect("Failed to add tensor");
            
            let mut metadata = Metadata::new();
            metadata.insert("key".to_string(), MetadataValue::String("value".to_string()));
            
            tensor_writer.write_to_file(metadata).expect("Failed to write");
        }
        
        // Read phase
        {
            let cursor = Cursor::new(&buffer);
            let mut reader = GGUFFileReader::new(cursor).expect("Failed to create reader");
            
            assert_eq!(reader.tensor_count(), 1);
            assert_eq!(reader.metadata().get_string("key"), Some("value"));
            
            let tensor_info = reader.get_tensor_info("test").unwrap();
            assert_eq!(tensor_info.tensor_type(), TensorType::F32);
            assert_eq!(tensor_info.shape().dimensions(), &[2, 2]);
            
            let data = reader.load_tensor_data("test").expect("Failed to load").unwrap();
            assert_eq!(data.len(), 16); // 4 * 4 bytes
            
            let floats: Vec<f32> = data.as_slice()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            assert_eq!(floats, vec![1.0, 2.0, 3.0, 4.0]);
        }
    }
}