//! Build an unquantized GGUF file in memory and read its payload back.

#[cfg(feature = "std")]
use gguf_rs_lib::{
    builder::GGUFBuilder, reader::GGUFFileReader, tensor::TensorData, GGUFError, Result,
};
#[cfg(feature = "std")]
use std::io::Cursor;

#[cfg(feature = "std")]
fn main() -> Result<()> {
    let values = [1.0_f32, 2.0, 3.0, 4.0];
    let expected_bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();

    let (bytes, write_result) = GGUFBuilder::simple("roundtrip", "In-memory example")
        .add_f32_tensor("weights", vec![2, 2], values.to_vec())?
        .build_to_bytes()?;

    let mut reader = GGUFFileReader::new(Cursor::new(bytes))?;
    let payload: TensorData = reader
        .load_tensor_data("weights")?
        .ok_or_else(|| GGUFError::Format("weights payload was not returned".to_string()))?;

    if payload.as_slice() != expected_bytes {
        return Err(GGUFError::InvalidTensorData(
            "round-trip payload differs from input".to_string(),
        ));
    }

    println!("wrote {} bytes", write_result.total_bytes_written);
    println!(
        "read back {} metadata entries, {} tensor, and {} payload bytes",
        reader.metadata().len(),
        reader.tensor_count(),
        payload.len()
    );

    Ok(())
}

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("roundtrip_test requires the default `std` feature");
    std::process::exit(1);
}
