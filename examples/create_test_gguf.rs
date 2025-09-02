//! Creates a simple test GGUF file for testing the library
//!
//! This example requires the `std` feature because it uses file I/O operations.

#[cfg(feature = "std")]
use gguf_rs::format::Metadata as FormatMetadata;
#[cfg(feature = "std")]
use gguf_rs::prelude::*;
#[cfg(feature = "std")]
use gguf_rs::tensor::{TensorData, TensorInfo, TensorShape, TensorType};

#[cfg(feature = "std")]
fn main() -> Result<()> {
    let output_path = "test_model.gguf";

    println!("Creating test GGUF file: {}", output_path);

    // Create metadata
    let mut metadata = FormatMetadata::new();
    metadata.insert("general.name".to_string(), MetadataValue::String("Test Model".to_string()));
    metadata.insert("general.architecture".to_string(), MetadataValue::String("test".to_string()));
    metadata.insert("general.version".to_string(), MetadataValue::U32(1));
    metadata.insert("general.parameter_count".to_string(), MetadataValue::U64(1000000));
    metadata.insert(
        "general.description".to_string(),
        MetadataValue::String("A simple test model for gguf_rs".to_string()),
    );

    // Create some test tensors
    let mut tensors = Vec::new();

    // Small embedding tensor (vocabulary embeddings)
    let vocab_shape = TensorShape::new(vec![1000, 128])?; // 1000 vocab, 128 dim
    let vocab_data = TensorData::new_owned(vec![0u8; 1000 * 128 * 4]); // F32 data
    let vocab_tensor =
        TensorInfo::new("token_embd.weight".to_string(), vocab_shape, TensorType::F32, 0);
    tensors.push((vocab_tensor, vocab_data));

    // Layer norm weights
    let norm_shape = TensorShape::new(vec![128])?;
    let norm_data = TensorData::new_owned(vec![1u8; 128 * 4]); // F32 ones
    let norm_tensor =
        TensorInfo::new("output_norm.weight".to_string(), norm_shape, TensorType::F32, 0);
    tensors.push((norm_tensor, norm_data));

    // Output projection
    let out_shape = TensorShape::new(vec![128, 1000])?;
    let out_data = TensorData::new_owned(vec![0u8; 128 * 1000 * 4]); // F32 data
    let out_tensor = TensorInfo::new("output.weight".to_string(), out_shape, TensorType::F32, 0);
    tensors.push((out_tensor, out_data));

    // Create the writer and write the file
    let file = std::fs::File::create(output_path)?;
    let mut writer = GGUFFileWriter::new(file);

    // Create header
    let header = GGUFHeader::new(tensors.len() as u64, metadata.len() as u64);
    writer.write_header(&header)?;

    // Write metadata
    writer.write_metadata(&metadata)?;

    // Write tensor infos
    let tensor_infos: Vec<TensorInfo> = tensors.iter().map(|(info, _)| info.clone()).collect();
    writer.write_tensor_infos(&tensor_infos)?;

    // Align for tensor data
    writer.align_for_tensor_data()?;

    // Write tensor data
    for (tensor_info, tensor_data) in &tensors {
        writer.write_tensor_data(tensor_info, tensor_data)?;
    }

    println!("Successfully created test GGUF file!");
    println!("File size: {} bytes", std::fs::metadata(output_path)?.len());

    // Verify by reading it back
    println!("\nVerifying the created file...");
    let read_file = std::fs::File::open(output_path)?;
    let reader = GGUFFileReader::new(read_file)?;

    println!("Verification successful!");
    println!("- Tensor count: {}", reader.tensor_infos().len());
    println!("- Metadata entries: {}", reader.metadata().len());

    Ok(())
}

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("This example requires the 'std' feature to be enabled.");
    eprintln!("Run with: cargo run --example create_test_gguf --features std");
    std::process::exit(1);
}
