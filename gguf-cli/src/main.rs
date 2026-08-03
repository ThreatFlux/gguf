//! Command-line inspection, validation, and comparison for GGUF files.

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
#[cfg(feature = "color")]
use colored::Colorize;
use gguf_rs_lib::{
    format::{GGUFTensorType, Metadata, MetadataValue},
    reader::GGUFFileReader,
    tensor::TensorInfo,
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "gguf-cli", version, about = "Inspect, validate, and compare GGUF files")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Print progress details.
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Suppress colored output.
    #[arg(long, global = true)]
    no_color: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Display header and high-level information.
    Info {
        /// Path to a GGUF file.
        file: PathBuf,

        /// Include file size, tensor bytes, and common metadata.
        #[arg(short, long)]
        detailed: bool,
    },

    /// List tensor descriptors.
    Tensors {
        /// Path to a GGUF file.
        file: PathBuf,

        /// Keep tensors whose name contains this string.
        #[arg(short, long)]
        filter: Option<String>,

        /// Print one compact line per tensor.
        #[arg(short, long)]
        summary: bool,
    },

    /// Display metadata in a deterministic order.
    Metadata {
        /// Path to a GGUF file.
        file: PathBuf,

        /// Output format.
        #[arg(short, long, value_enum, default_value_t = MetadataFormat::Table)]
        format: MetadataFormat,

        /// Keep metadata keys containing this string.
        #[arg(short, long)]
        key: Option<String>,
    },

    /// Parse one file or a directory of .gguf files.
    Validate {
        /// Path to a GGUF file or directory.
        path: PathBuf,

        /// Read every declared tensor payload after structural validation.
        #[arg(short, long)]
        integrity: bool,

        /// Recurse into subdirectories; valid only for a directory path.
        #[arg(short, long)]
        recursive: bool,
    },

    /// Compare metadata and tensor descriptors from two GGUF files.
    Compare {
        /// First GGUF file.
        file1: PathBuf,

        /// Second GGUF file.
        file2: PathBuf,

        /// Also compare every matching tensor payload.
        #[arg(short, long)]
        data: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MetadataFormat {
    Json,
    Yaml,
    Toml,
    Table,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    #[cfg(feature = "color")]
    if cli.no_color {
        colored::control::set_override(false);
    }
    #[cfg(not(feature = "color"))]
    let _ = cli.no_color;

    match cli.command {
        Commands::Info { file, detailed } => info_command(&file, detailed, cli.verbose),
        Commands::Tensors { file, filter, summary } => {
            tensors_command(&file, filter.as_deref(), summary, cli.verbose)
        }
        Commands::Metadata { file, format, key } => {
            metadata_command(&file, format, key.as_deref(), cli.verbose)
        }
        Commands::Validate { path, integrity, recursive } => {
            validate_command(&path, integrity, recursive, cli.verbose)
        }
        Commands::Compare { file1, file2, data } => {
            compare_command(&file1, &file2, data, cli.verbose)
        }
    }
}

fn info_command(file: &Path, detailed: bool, verbose: bool) -> Result<()> {
    if verbose {
        println!("Reading GGUF file: {}", file.display());
    }

    let reader = open_reader(file)?;

    println!("GGUF File Information");
    println!("=====================");
    println!("File: {}", file.display());
    println!("GGUF version: {}", reader.header().version);
    println!("Tensors: {}", reader.tensor_count());
    println!("Metadata entries: {}", reader.metadata().len());
    println!("Tensor alignment: {} bytes", reader.tensor_alignment());

    if detailed {
        println!("File size: {} bytes", std::fs::metadata(file)?.len());
        println!("Tensor payload bytes: {}", total_tensor_bytes(reader.tensor_infos())?);

        if let Some(name) = reader.metadata().get("general.name") {
            println!("Model name: {name}");
        }
        if let Some(architecture) = reader.metadata().get("general.architecture") {
            println!("Architecture: {architecture}");
        }
    }

    Ok(())
}

fn tensors_command(file: &Path, filter: Option<&str>, summary: bool, verbose: bool) -> Result<()> {
    if verbose {
        println!("Reading tensors from: {}", file.display());
    }

    let reader = open_reader(file)?;
    let tensors: Vec<_> = reader
        .tensor_infos()
        .iter()
        .filter(|tensor| filter.is_none_or(|pattern| tensor.name().contains(pattern)))
        .collect();

    println!("Found {} tensors", tensors.len());
    for tensor in tensors {
        if summary {
            println!(
                "{}: {} {:?}",
                tensor.name(),
                tensor.tensor_type().name(),
                tensor.shape().dims()
            );
        } else {
            println!("Tensor: {}", tensor.name());
            println!("  Type: {}", tensor.tensor_type().name());
            println!("  Shape: {:?}", tensor.shape().dims());
            println!("  Elements: {}", tensor.element_count());
            println!("  Payload bytes: {}", tensor.checked_expected_data_size()?);
            println!("  Relative offset: {}", tensor.data_offset());
        }
    }

    Ok(())
}

fn metadata_command(
    file: &Path,
    format: MetadataFormat,
    key_filter: Option<&str>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Reading metadata from: {}", file.display());
    }

    let reader = open_reader(file)?;
    let metadata = filtered_metadata(reader.metadata(), key_filter);

    match format {
        MetadataFormat::Json => {
            let output = machine_metadata(&metadata, false)?;
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        MetadataFormat::Yaml => {
            let output = machine_metadata(&metadata, true)?;
            print!("{}", serde_yaml::to_string(&output)?);
        }
        MetadataFormat::Toml => {
            let output = machine_metadata(&metadata, false)?;
            print!(
                "{}",
                toml::to_string_pretty(&output)
                    .context("metadata cannot be represented as TOML")?
            );
        }
        MetadataFormat::Table => {
            for (key, value) in metadata {
                println!("{key}: {value}");
            }
        }
    }

    Ok(())
}

fn validate_command(path: &Path, integrity: bool, recursive: bool, verbose: bool) -> Result<()> {
    if verbose {
        println!("Validating: {}", path.display());
    }

    if path.is_dir() {
        return validate_directory(path, integrity, recursive, verbose);
    }
    if recursive {
        bail!("--recursive requires a directory path");
    }

    match validate_file(path, integrity) {
        Ok(()) => {
            print_valid(path);
            Ok(())
        }
        Err(error) => {
            print_invalid(path, &error);
            Err(error)
        }
    }
}

fn validate_directory(
    directory: &Path,
    integrity: bool,
    recursive: bool,
    verbose: bool,
) -> Result<()> {
    let mut walker = WalkDir::new(directory).min_depth(1);
    if !recursive {
        walker = walker.max_depth(1);
    }

    let mut files = Vec::new();
    for entry in walker {
        let entry = entry.with_context(|| {
            format!("failed while traversing directory {}", directory.display())
        })?;
        if entry.file_type().is_file() && is_gguf_path(entry.path()) {
            files.push(entry.into_path());
        }
    }
    files.sort();

    if files.is_empty() {
        bail!("no .gguf files found in {}", directory.display());
    }

    let total = files.len();
    let mut failures = 0usize;
    for file in files {
        if verbose {
            println!("Parsing {}", file.display());
        }
        match validate_file(&file, integrity) {
            Ok(()) => print_valid(&file),
            Err(error) => {
                failures += 1;
                print_invalid(&file, &error);
            }
        }
    }

    if failures > 0 {
        bail!("{failures}/{total} GGUF files failed validation");
    }

    println!("Validated {total} GGUF files");
    Ok(())
}

fn validate_file(path: &Path, integrity: bool) -> Result<()> {
    let mut reader = open_reader(path)?;
    if integrity {
        reader.validate_all_tensor_data().with_context(|| {
            format!("failed to read all tensor payloads from {}", path.display())
        })?;
    }
    Ok(())
}

fn is_gguf_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

fn print_valid(path: &Path) {
    #[cfg(feature = "color")]
    println!("{}: {}", path.display(), "VALID".green());

    #[cfg(not(feature = "color"))]
    println!("{}: VALID", path.display());
}

fn print_invalid(path: &Path, error: &anyhow::Error) {
    #[cfg(feature = "color")]
    println!("{}: {} - {error:#}", path.display(), "INVALID".red());

    #[cfg(not(feature = "color"))]
    println!("{}: INVALID - {error:#}", path.display());
}

fn compare_command(file1: &Path, file2: &Path, compare_data: bool, verbose: bool) -> Result<()> {
    if verbose {
        println!("Comparing {} with {}", file1.display(), file2.display());
    }

    let mut left = open_reader(file1)?;
    let mut right = open_reader(file2)?;
    let mut differences = Vec::new();

    if left.header().version != right.header().version {
        differences.push(format!(
            "GGUF version differs: left={}, right={}",
            left.header().version,
            right.header().version
        ));
    }

    let metadata_keys: BTreeSet<String> = left
        .metadata()
        .iter()
        .map(|(key, _)| key.clone())
        .chain(right.metadata().iter().map(|(key, _)| key.clone()))
        .collect();
    for key in metadata_keys {
        let left_value = left.metadata().get(&key);
        let right_value = right.metadata().get(&key);
        if !metadata_values_equal(left_value, right_value) {
            differences.push(format!(
                "metadata differs at {key}: left={}, right={}",
                summarize_metadata(left_value),
                summarize_metadata(right_value)
            ));
        }
    }

    let left_descriptors = tensor_descriptors(&left)?;
    let right_descriptors = tensor_descriptors(&right)?;
    let tensor_names: BTreeSet<String> = left_descriptors
        .keys()
        .cloned()
        .chain(right_descriptors.keys().cloned())
        .collect();

    for name in &tensor_names {
        let left_descriptor = left_descriptors.get(name);
        let right_descriptor = right_descriptors.get(name);
        if left_descriptor != right_descriptor {
            differences.push(format!(
                "tensor descriptor differs at {name}: \
                 left={left_descriptor:?}, right={right_descriptor:?}"
            ));
        }
    }

    if compare_data {
        for name in tensor_names {
            let Some(left_descriptor) = left_descriptors.get(&name) else {
                continue;
            };
            let Some(right_descriptor) = right_descriptors.get(&name) else {
                continue;
            };
            if left_descriptor != right_descriptor {
                continue;
            }

            let equal = left.tensor_data_equals(&name, &mut right).with_context(|| {
                format!(
                    "failed to compare tensor {name} from {} and {}",
                    file1.display(),
                    file2.display()
                )
            })?;
            if !equal {
                differences.push(format!("tensor payload differs at {name}"));
            }
        }
    }

    if differences.is_empty() {
        if compare_data {
            println!("Files have identical metadata, tensor descriptors, and tensor payloads");
        } else {
            println!("Files have identical metadata and tensor descriptors");
        }
        return Ok(());
    }

    for difference in &differences {
        println!("DIFF: {difference}");
    }
    let count = differences.len();
    if count == 1 {
        bail!("files differ: 1 difference found");
    }
    bail!("files differ: {count} differences found")
}

fn metadata_values_equal(left: Option<&MetadataValue>, right: Option<&MetadataValue>) -> bool {
    match (left, right) {
        (Some(MetadataValue::F32(left)), Some(MetadataValue::F32(right))) => {
            left.to_bits() == right.to_bits()
        }
        (Some(MetadataValue::F64(left)), Some(MetadataValue::F64(right))) => {
            left.to_bits() == right.to_bits()
        }
        (Some(MetadataValue::Array(left)), Some(MetadataValue::Array(right))) => {
            left.element_type == right.element_type
                && left.length == right.length
                && left.values.len() == right.values.len()
                && left
                    .values
                    .iter()
                    .zip(&right.values)
                    .all(|(left, right)| metadata_values_equal(Some(left), Some(right)))
        }
        _ => left == right,
    }
}

fn open_reader(path: &Path) -> Result<GGUFFileReader<File>> {
    let file =
        File::open(path).with_context(|| format!("failed to open GGUF file {}", path.display()))?;
    GGUFFileReader::new(file)
        .with_context(|| format!("failed to parse GGUF file {}", path.display()))
}

fn total_tensor_bytes(tensors: &[TensorInfo]) -> Result<u64> {
    let mut total = 0u64;
    for tensor in tensors {
        total = total
            .checked_add(tensor.checked_expected_data_size()?)
            .ok_or_else(|| anyhow!("total tensor payload size overflows u64"))?;
    }
    Ok(total)
}

fn filtered_metadata<'a>(
    metadata: &'a Metadata,
    key_filter: Option<&str>,
) -> BTreeMap<&'a str, &'a MetadataValue> {
    metadata
        .iter()
        .filter(|(key, _)| key_filter.is_none_or(|pattern| key.contains(pattern)))
        .map(|(key, value)| (key.as_str(), value))
        .collect()
}

#[derive(Serialize)]
#[serde(untagged)]
enum MachineValue<'a> {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(&'a str),
    Array(Vec<MachineValue<'a>>),
}

fn machine_metadata<'a>(
    metadata: &BTreeMap<&'a str, &'a MetadataValue>,
    allow_non_finite: bool,
) -> Result<BTreeMap<&'a str, MachineValue<'a>>> {
    metadata
        .iter()
        .map(|(key, value)| {
            machine_value(value, allow_non_finite)
                .with_context(|| format!("metadata value {key} cannot be serialized"))
                .map(|value| (*key, value))
        })
        .collect()
}

fn machine_value(value: &MetadataValue, allow_non_finite: bool) -> Result<MachineValue<'_>> {
    let output = match value {
        MetadataValue::U8(value) => MachineValue::Unsigned(u64::from(*value)),
        MetadataValue::I8(value) => MachineValue::Signed(i64::from(*value)),
        MetadataValue::U16(value) => MachineValue::Unsigned(u64::from(*value)),
        MetadataValue::I16(value) => MachineValue::Signed(i64::from(*value)),
        MetadataValue::U32(value) => MachineValue::Unsigned(u64::from(*value)),
        MetadataValue::I32(value) => MachineValue::Signed(i64::from(*value)),
        MetadataValue::F32(value) => {
            let value = f64::from(*value);
            if !allow_non_finite && !value.is_finite() {
                bail!("non-finite floating-point value is not supported by this output format");
            }
            MachineValue::Float(value)
        }
        MetadataValue::Bool(value) => MachineValue::Bool(*value),
        MetadataValue::String(value) => MachineValue::String(value),
        MetadataValue::Array(array) => {
            let mut values = Vec::new();
            values
                .try_reserve_exact(array.values.len())
                .map_err(|_| anyhow!("unable to allocate metadata output array"))?;
            for value in &array.values {
                values.push(machine_value(value, allow_non_finite)?);
            }
            MachineValue::Array(values)
        }
        MetadataValue::U64(value) => MachineValue::Unsigned(*value),
        MetadataValue::I64(value) => MachineValue::Signed(*value),
        MetadataValue::F64(value) => {
            if !allow_non_finite && !value.is_finite() {
                bail!("non-finite floating-point value is not supported by this output format");
            }
            MachineValue::Float(*value)
        }
    };
    Ok(output)
}

fn summarize_metadata(value: Option<&MetadataValue>) -> String {
    const PREVIEW_CHARS: usize = 80;

    match value {
        None => "<missing>".to_string(),
        Some(MetadataValue::String(value)) => {
            let mut preview: String = value.chars().take(PREVIEW_CHARS).collect();
            if preview.len() < value.len() {
                preview.push('…');
            }
            format!("string({} bytes, {preview:?})", value.len())
        }
        Some(MetadataValue::Array(array)) => {
            format!("array(type={}, len={})", array.element_type, array.length)
        }
        Some(value) => value.to_string_representation(),
    }
}

#[derive(Debug, PartialEq, Eq)]
struct TensorDescriptor {
    tensor_type: GGUFTensorType,
    dimensions: Vec<u64>,
    relative_offset: u64,
    payload_bytes: u64,
}

fn tensor_descriptors(reader: &GGUFFileReader<File>) -> Result<BTreeMap<String, TensorDescriptor>> {
    reader
        .tensor_infos()
        .iter()
        .map(|tensor| {
            Ok((
                tensor.name().to_string(),
                TensorDescriptor {
                    tensor_type: tensor.tensor_type(),
                    dimensions: tensor.shape().dims().to_vec(),
                    relative_offset: tensor.data_offset(),
                    payload_bytes: tensor.checked_expected_data_size()?,
                },
            ))
        })
        .collect()
}
