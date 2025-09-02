//! # gguf-cli
//!
//! Command-line tool for working with GGUF (GGML Universal Format) files.
//!
//! This tool provides various subcommands for inspecting, validating, and manipulating
//! GGUF files from the command line.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use gguf_rs_lib::prelude::*;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "gguf-cli",
    version,
    about = "Command-line tool for working with GGUF files",
    long_about = "A comprehensive command-line tool for inspecting, validating, and manipulating GGUF (GGML Universal Format) files."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Suppress colored output
    #[arg(long, global = true)]
    no_color: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Display information about a GGUF file
    Info {
        /// Path to the GGUF file
        file: PathBuf,

        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// List all tensors in a GGUF file
    Tensors {
        /// Path to the GGUF file
        file: PathBuf,

        /// Filter tensors by name pattern
        #[arg(short, long)]
        filter: Option<String>,

        /// Show tensor data types and shapes only
        #[arg(short, long)]
        summary: bool,
    },

    /// Display metadata from a GGUF file
    Metadata {
        /// Path to the GGUF file
        file: PathBuf,

        /// Output format (json, yaml, toml, table)
        #[arg(short, long, default_value = "table")]
        format: String,

        /// Filter metadata by key pattern
        #[arg(short, long)]
        key: Option<String>,
    },

    /// Validate a GGUF file
    Validate {
        /// Path to the GGUF file or directory
        path: PathBuf,

        /// Check file integrity
        #[arg(short, long)]
        integrity: bool,

        /// Recursive validation for directories
        #[arg(short, long)]
        recursive: bool,
    },

    /// Compare two GGUF files
    Compare {
        /// First GGUF file
        file1: PathBuf,

        /// Second GGUF file
        file2: PathBuf,

        /// Compare tensor data (slower)
        #[arg(short, long)]
        data: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize colored output
    #[cfg(feature = "color")]
    if cli.no_color {
        colored::control::set_override(false);
    }

    match cli.command {
        Commands::Info { file, detailed } => info_command(&file, detailed, cli.verbose),
        Commands::Tensors { file, filter, summary } => {
            tensors_command(&file, filter.as_deref(), summary, cli.verbose)
        }
        Commands::Metadata { file, format, key } => {
            metadata_command(&file, &format, key.as_deref(), cli.verbose)
        }
        Commands::Validate { path, integrity, recursive } => {
            validate_command(&path, integrity, recursive, cli.verbose)
        }
        Commands::Compare { file1, file2, data } => {
            compare_command(&file1, &file2, data, cli.verbose)
        }
    }
}

fn info_command(file: &PathBuf, detailed: bool, verbose: bool) -> Result<()> {
    if verbose {
        println!("Reading GGUF file: {}", file.display());
    }

    let gguf_file = std::fs::File::open(file)
        .with_context(|| format!("Failed to open file: {}", file.display()))?;

    let gguf = GGUFFileReader::new(gguf_file).with_context(|| "Failed to parse GGUF file")?;

    // Basic information
    println!("GGUF File Information");
    println!("=====================");
    println!("File: {}", file.display());
    println!("GGUF Version: {}", gguf.header().version);
    println!("Number of tensors: {}", gguf.tensor_count());
    println!("Number of metadata entries: {}", gguf.metadata().len());

    if detailed {
        println!("\nFile size: {} bytes", std::fs::metadata(file)?.len());

        // Calculate total tensor size
        let total_tensor_bytes: u64 =
            gguf.tensor_infos().iter().map(|t| t.expected_data_size()).sum();
        println!("Total tensor data: {} bytes", total_tensor_bytes);

        // Show some key metadata if available
        if let Some(name) = gguf.metadata().get("general.name") {
            println!("Model name: {}", name);
        }
        if let Some(architecture) = gguf.metadata().get("general.architecture") {
            println!("Architecture: {}", architecture);
        }
    }

    Ok(())
}

fn tensors_command(
    file: &PathBuf,
    filter: Option<&str>,
    summary: bool,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Reading tensors from: {}", file.display());
    }

    let gguf_file = std::fs::File::open(file)
        .with_context(|| format!("Failed to open file: {}", file.display()))?;

    let gguf = GGUFFileReader::new(gguf_file).with_context(|| "Failed to parse GGUF file")?;

    let tensors: Vec<_> = if let Some(pattern) = filter {
        gguf.tensor_infos().iter().filter(|t| t.name().contains(pattern)).collect()
    } else {
        gguf.tensor_infos().iter().collect()
    };

    if summary {
        println!("Found {} tensors", tensors.len());
        for tensor in tensors {
            println!("{}: {:?} {:?}", tensor.name(), tensor.tensor_type(), tensor.shape());
        }
    } else {
        // Detailed tensor information would go here
        println!("Detailed tensor listing not yet implemented");
    }

    Ok(())
}

fn metadata_command(
    file: &PathBuf,
    format: &str,
    key_filter: Option<&str>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Reading metadata from: {}", file.display());
    }

    let gguf_file = std::fs::File::open(file)
        .with_context(|| format!("Failed to open file: {}", file.display()))?;

    let gguf = GGUFFileReader::new(gguf_file).with_context(|| "Failed to parse GGUF file")?;

    let metadata: Vec<_> = if let Some(pattern) = key_filter {
        gguf.metadata().iter().filter(|(k, _)| k.contains(pattern)).collect()
    } else {
        gguf.metadata().iter().collect()
    };

    match format {
        "json" => {
            let json_value = serde_json::to_value(&metadata)?;
            println!("{}", serde_json::to_string_pretty(&json_value)?);
        }
        "yaml" => {
            println!("{}", serde_yaml::to_string(&metadata)?);
        }
        "toml" => {
            println!("{}", toml::to_string_pretty(&metadata)?);
        }
        "table" | _ => {
            println!("Metadata (table format not yet implemented)");
            for (key, value) in metadata {
                println!("{}: {:?}", key, value);
            }
        }
    }

    Ok(())
}

fn validate_command(
    path: &PathBuf,
    _integrity: bool,
    _recursive: bool,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Validating: {}", path.display());
    }

    if path.is_dir() {
        println!("Directory validation not yet implemented");
        return Ok(());
    }

    let gguf_file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;

    match GGUFFileReader::new(gguf_file) {
        Ok(_) => {
            #[cfg(feature = "color")]
            println!("{}: {}", path.display(), colored::Colorize::green("VALID"));

            #[cfg(not(feature = "color"))]
            println!("{}: VALID", path.display());
        }
        Err(e) => {
            #[cfg(feature = "color")]
            println!("{}: {} - {}", path.display(), colored::Colorize::red("INVALID"), e);

            #[cfg(not(feature = "color"))]
            println!("{}: INVALID - {}", path.display(), e);

            return Err(e.into());
        }
    }

    Ok(())
}

fn compare_command(
    file1: &PathBuf,
    file2: &PathBuf,
    _compare_data: bool,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Comparing: {} vs {}", file1.display(), file2.display());
    }

    println!("File comparison not yet implemented");
    Ok(())
}
