//! Integration test for basic GGUF functionality

#[cfg(all(test, feature = "std"))]
mod tests {
    use crate::prelude::*;
    use std::io::Cursor;

    #[test]
    fn test_basic_gguf_functionality() {
        // Test creating basic data structures
        let metadata = Metadata::new();
        assert_eq!(metadata.len(), 0);

        let shape = TensorShape::new(vec![2, 3]).unwrap();
        assert_eq!(shape.element_count(), 6);

        let data = TensorData::new_owned(vec![0u8; 24]);
        assert_eq!(data.len(), 24);

        let tensor_info = TensorInfo::new("test".to_string(), shape, TensorType::F32, 0);
        assert_eq!(tensor_info.name(), "test");
        assert_eq!(tensor_info.element_count(), 6);
        assert_eq!(tensor_info.expected_data_size(), 24);

        println!("✅ Basic GGUF data structures work correctly");
    }

    #[test]
    fn test_metadata_operations() {
        let mut metadata = Metadata::new();
        
        metadata.insert("name".to_string(), MetadataValue::String("test".to_string()));
        metadata.insert("version".to_string(), MetadataValue::U32(42));
        metadata.insert("temperature".to_string(), MetadataValue::F32(0.8));
        
        assert_eq!(metadata.len(), 3);
        assert_eq!(metadata.get_string("name"), Some("test"));
        assert_eq!(metadata.get_u64("version"), Some(42));
        assert_eq!(metadata.get_f64("temperature"), Some(0.8f64));

        println!("✅ Metadata operations work correctly");
    }

    #[test]
    fn test_tensor_shape_operations() {
        let shape = TensorShape::new(vec![2, 3, 4]).unwrap();
        
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.element_count(), 24);
        assert_eq!(shape.dims(), &[2, 3, 4]);
        
        let reshaped = shape.reshape(vec![6, 4]).unwrap();
        assert_eq!(reshaped.dims(), &[6, 4]);
        assert_eq!(reshaped.element_count(), 24);

        let flattened = shape.flatten();
        assert_eq!(flattened.dims(), &[24]);

        println!("✅ TensorShape operations work correctly");
    }

    #[test]
    fn test_minimal_stream_writer() {
        let mut buffer = Vec::new();
        
        // Create minimal GGUF
        let metadata = Metadata::new();
        let tensors = Vec::<(TensorInfo, TensorData)>::new();

        let cursor = Cursor::new(&mut buffer);
        let mut writer = crate::writer::stream_writer::GGUFStreamWriter::new(cursor);
        
        let result = writer.write_complete_stream(&metadata, &tensors);
        assert!(result.is_ok(), "Stream writer should work");
        
        assert!(!buffer.is_empty(), "Buffer should contain data");
        println!("✅ Minimal stream writer works: {} bytes", buffer.len());
    }

    #[test] 
    fn test_minimal_stream_reader() {
        // Create minimal GGUF data manually
        let mut buffer = Vec::new();
        
        // Magic number (GGUF)
        buffer.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        // Version 3
        buffer.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count (0)
        buffer.extend_from_slice(&0u64.to_le_bytes());
        // Metadata count (0)
        buffer.extend_from_slice(&0u64.to_le_bytes());
        
        // Add padding to 32-byte alignment
        while buffer.len() % 32 != 0 {
            buffer.push(0);
        }
        
        let cursor = Cursor::new(buffer);
        let reader = crate::reader::stream_reader::GGUFStreamReader::new(cursor);
        
        assert!(reader.is_ok(), "Stream reader should work");
        let reader = reader.unwrap();
        
        assert_eq!(reader.header().version, 3);
        assert_eq!(reader.tensor_count(), 0);
        assert_eq!(reader.metadata().len(), 0);
        
        println!("✅ Minimal stream reader works");
    }
}