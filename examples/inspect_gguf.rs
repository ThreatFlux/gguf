//! Inspect GGUF v3 metadata and tensor descriptors.
//!
//! Payload byte sizes use checked scalar widths or GGML block layouts for the
//! tensor types accepted by the current crate.

#[cfg(feature = "std")]
use gguf_rs_lib::reader::GGUFFileReader;
#[cfg(feature = "std")]
use std::{error::Error, fs::File, path::PathBuf};

#[cfg(feature = "std")]
fn run() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: inspect_gguf <path-to-gguf>")?;

    let file_size = std::fs::metadata(&path)?.len();
    let reader = GGUFFileReader::new(File::open(&path)?)?;

    println!("file: {}", path.display());
    println!("file bytes: {file_size}");
    println!("GGUF version: {}", reader.header().version);
    println!("metadata entries: {}", reader.metadata().len());
    println!("tensor descriptors: {}", reader.tensor_count());

    let mut metadata: Vec<_> = reader.metadata().iter().collect();
    metadata.sort_by_key(|(left, _)| *left);

    println!("\nmetadata:");
    for (key, value) in metadata {
        println!("- {key}: {value}");
    }

    println!("\ntensors:");
    for tensor in reader.tensor_infos() {
        let tensor_type = tensor.tensor_type();
        println!(
            "- {}: {} {:?}, offset {}",
            tensor.name(),
            tensor_type.name(),
            tensor.shape().dims(),
            tensor.data_offset()
        );

        println!("  payload bytes: {}", tensor.checked_expected_data_size()?);
    }

    println!(
        "\ninspection parsed descriptors only; tensor payloads were not loaded or authenticated"
    );

    Ok(())
}

#[cfg(feature = "std")]
fn main() {
    if let Err(error) = run() {
        eprintln!("inspect_gguf: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("inspect_gguf requires the default `std` feature");
    std::process::exit(1);
}
