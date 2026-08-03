//! Read GGUF v3 metadata and tensor descriptors from a seekable file.
//!
//! Run with:
//! `cargo run --example basic_usage -- path/to/model.gguf`

#[cfg(feature = "std")]
use gguf_rs_lib::reader::GGUFFileReader;
#[cfg(feature = "std")]
use std::{error::Error, fs::File, path::PathBuf};

#[cfg(feature = "std")]
fn run() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: basic_usage <path-to-gguf>")?;

    let reader = GGUFFileReader::new(File::open(&path)?)?;

    println!("file: {}", path.display());
    println!("GGUF version: {}", reader.header().version);
    println!("metadata entries: {}", reader.metadata().len());
    println!("tensor descriptors: {}", reader.tensor_count());

    if let Some(name) = reader.metadata().get_string("general.name") {
        println!("model name: {name}");
    }

    println!("\nfirst tensor descriptors:");
    for tensor in reader.tensor_infos().iter().take(10) {
        println!(
            "- {}: {} {:?}",
            tensor.name(),
            tensor.tensor_type().name(),
            tensor.shape().dims()
        );
    }

    let remaining = reader.tensor_count().saturating_sub(10);
    if remaining > 0 {
        println!("... and {remaining} more");
    }

    Ok(())
}

#[cfg(feature = "std")]
fn main() {
    if let Err(error) = run() {
        eprintln!("basic_usage: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("basic_usage requires the default `std` feature");
    std::process::exit(1);
}
