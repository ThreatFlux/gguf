//! Validate a GGUF magic/version prefix with the async preview API.
//!
//! This API does not yet parse metadata, tensor descriptors, or payloads.
//! Run with:
//! `cargo run --example async_usage --features async -- path/to/model.gguf`

#[cfg(feature = "async")]
use gguf_rs_lib::{r#async::AsyncGGUFFile, GGUFError, Result};
#[cfg(feature = "async")]
use std::path::PathBuf;

#[cfg(feature = "async")]
#[tokio::main]
async fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| GGUFError::Format("usage: async_usage <path-to-gguf>".to_string()))?;

    let header = AsyncGGUFFile::read_file_async(&path).await?;

    println!("file: {}", path.display());
    println!("validated GGUF v{} magic/version prefix", header.version);
    println!("metadata and tensor parsing are not implemented by this preview API");

    Ok(())
}

#[cfg(not(feature = "async"))]
fn main() {
    eprintln!("async_usage requires `--features async`");
    std::process::exit(1);
}
