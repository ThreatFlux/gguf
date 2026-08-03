use assert_cmd::cargo::cargo_bin_cmd;
use gguf_rs_lib::{builder::GGUFBuilder, format::MetadataValue};
use predicates::prelude::*;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn write_fixture(directory: &Path, file_name: &str, weight: f32) -> PathBuf {
    let path = directory.join(file_name);
    GGUFBuilder::simple("test-model", "CLI integration fixture")
        .add_metadata("general.architecture", MetadataValue::String("test".to_string()))
        .add_f32_tensor("weights", vec![2, 2], vec![weight; 4])
        .expect("valid F32 fixture")
        .build_to_file(&path)
        .expect("fixture should be written");
    path
}

#[test]
fn info_and_tensor_views_report_real_data() {
    let directory = TempDir::new().expect("temporary directory");
    let fixture = write_fixture(directory.path(), "model.gguf", 1.0);

    cargo_bin_cmd!("gguf-cli")
        .args(["--no-color", "info"])
        .arg(&fixture)
        .arg("--detailed")
        .assert()
        .success()
        .stdout(
            predicate::str::contains("GGUF version: 3")
                .and(predicate::str::contains("Tensor payload bytes: 16"))
                .and(predicate::str::contains("Model name: test-model")),
        );

    cargo_bin_cmd!("gguf-cli")
        .args(["--no-color", "tensors"])
        .arg(&fixture)
        .assert()
        .success()
        .stdout(
            predicate::str::contains("Tensor: weights")
                .and(predicate::str::contains("Type: F32"))
                .and(predicate::str::contains("Payload bytes: 16"))
                .and(predicate::str::contains("not yet implemented").not()),
        );
}

#[test]
fn metadata_formats_are_explicit_and_invalid_values_fail() {
    let directory = TempDir::new().expect("temporary directory");
    let fixture = write_fixture(directory.path(), "model.gguf", 1.0);

    cargo_bin_cmd!("gguf-cli")
        .args(["--no-color", "metadata"])
        .arg(&fixture)
        .args(["--format", "table", "--key", "general.name"])
        .assert()
        .success()
        .stdout(predicate::str::contains("general.name: test-model"));

    cargo_bin_cmd!("gguf-cli")
        .args(["--no-color", "metadata"])
        .arg(&fixture)
        .args(["--format", "json"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains(r#""general.name": "test-model""#)
                .and(predicate::str::contains(r#""String""#).not()),
        );

    cargo_bin_cmd!("gguf-cli")
        .args(["--no-color", "metadata"])
        .arg(&fixture)
        .args(["--format", "yaml"])
        .assert()
        .success()
        .stdout(predicate::str::contains("general.name: test-model"));

    cargo_bin_cmd!("gguf-cli")
        .args(["--no-color", "metadata"])
        .arg(&fixture)
        .args(["--format", "toml"])
        .assert()
        .success()
        .stdout(predicate::str::contains(r#""general.name" = "test-model""#));

    cargo_bin_cmd!("gguf-cli")
        .args(["metadata"])
        .arg(&fixture)
        .args(["--format", "csv"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid value 'csv'"));

    let non_finite = directory.path().join("non-finite.gguf");
    GGUFBuilder::simple("non-finite", "CLI serialization fixture")
        .add_metadata("test.value", MetadataValue::F64(f64::NAN))
        .build_to_file(&non_finite)
        .expect("non-finite fixture should be written");

    cargo_bin_cmd!("gguf-cli")
        .args(["metadata"])
        .arg(&non_finite)
        .args(["--format", "json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("non-finite floating-point value is not supported"));

    cargo_bin_cmd!("gguf-cli")
        .args(["--no-color", "compare"])
        .arg(&non_finite)
        .arg(&non_finite)
        .assert()
        .success()
        .stdout(predicate::str::contains("identical metadata and tensor descriptors"));
}

#[test]
fn recursive_validation_reports_invalid_files_and_fails() {
    let directory = TempDir::new().expect("temporary directory");
    let nested = directory.path().join("nested");
    std::fs::create_dir(&nested).expect("nested directory");
    let valid = write_fixture(&nested, "valid.gguf", 1.0);
    let invalid = directory.path().join("invalid.gguf");
    std::fs::write(&invalid, b"not a GGUF file").expect("invalid fixture");

    cargo_bin_cmd!("gguf-cli")
        .args(["--no-color", "validate"])
        .arg(directory.path())
        .arg("--recursive")
        .assert()
        .failure()
        .stdout(
            predicate::str::contains(format!("{}: VALID", valid.display()))
                .and(predicate::str::contains(format!("{}: INVALID", invalid.display()))),
        )
        .stderr(predicate::str::contains("1/2 GGUF files failed validation"));
}

#[test]
fn validation_reads_payloads_and_rejects_empty_directories() {
    let directory = TempDir::new().expect("temporary directory");
    let fixture = write_fixture(directory.path(), "model.gguf", 1.0);

    cargo_bin_cmd!("gguf-cli")
        .args(["--no-color", "validate"])
        .arg(&fixture)
        .arg("--integrity")
        .assert()
        .success()
        .stdout(predicate::str::contains("VALID"));

    let truncated = write_fixture(directory.path(), "truncated.gguf", 1.0);
    let truncated_length = std::fs::metadata(&truncated)
        .expect("fixture metadata")
        .len()
        .checked_sub(1)
        .expect("fixture is not empty");
    std::fs::OpenOptions::new()
        .write(true)
        .open(&truncated)
        .expect("open fixture for truncation")
        .set_len(truncated_length)
        .expect("truncate tensor payload");

    cargo_bin_cmd!("gguf-cli")
        .args(["--no-color", "validate"])
        .arg(&truncated)
        .arg("--integrity")
        .assert()
        .failure()
        .stdout(predicate::str::contains("INVALID"))
        .stderr(predicate::str::contains("Unexpected end of file"));

    let empty = TempDir::new().expect("empty temporary directory");
    cargo_bin_cmd!("gguf-cli")
        .args(["validate"])
        .arg(empty.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains("no .gguf files found"));
}

#[test]
fn compare_distinguishes_structure_from_payload_data() {
    let directory = TempDir::new().expect("temporary directory");
    let left = write_fixture(directory.path(), "left.gguf", 1.0);
    let right = write_fixture(directory.path(), "right.gguf", 2.0);

    cargo_bin_cmd!("gguf-cli")
        .args(["--no-color", "compare"])
        .arg(&left)
        .arg(&right)
        .assert()
        .success()
        .stdout(predicate::str::contains("identical metadata and tensor descriptors"));

    cargo_bin_cmd!("gguf-cli")
        .args(["--no-color", "compare"])
        .arg(&left)
        .arg(&right)
        .arg("--data")
        .assert()
        .failure()
        .stdout(predicate::str::contains("DIFF: tensor payload differs at weights"))
        .stderr(predicate::str::contains("files differ: 1 difference found"));

    let structurally_different = directory.path().join("different.gguf");
    GGUFBuilder::simple("different-model", "CLI integration fixture")
        .add_metadata("general.architecture", MetadataValue::String("test".to_string()))
        .add_f32_tensor("weights", vec![2, 2], vec![1.0; 4])
        .expect("valid F32 fixture")
        .build_to_file(&structurally_different)
        .expect("fixture should be written");

    cargo_bin_cmd!("gguf-cli")
        .args(["--no-color", "compare"])
        .arg(&left)
        .arg(&structurally_different)
        .assert()
        .failure()
        .stdout(predicate::str::contains("DIFF: metadata differs at general.name"))
        .stderr(predicate::str::contains("files differ: 1 difference found"));
}
