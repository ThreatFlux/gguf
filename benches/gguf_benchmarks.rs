use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gguf::format::Metadata as FormatMetadata;
use gguf::prelude::*;
use gguf::reader::file_reader::GGUFFileReader;
use gguf::tensor::{TensorData, TensorType};
use std::io::Cursor;

// Create some test GGUF data for benchmarking
fn create_test_gguf_data() -> Vec<u8> {
    let mut data = Vec::new();

    // GGUF magic and version
    data.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF magic
    data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count
    data.extend_from_slice(&0u64.to_le_bytes()); // Metadata count

    data
}

fn benchmark_gguf_read(c: &mut Criterion) {
    let test_data = create_test_gguf_data();

    c.bench_function("gguf_read_minimal", |b| {
        b.iter(|| {
            let cursor = Cursor::new(black_box(&test_data));
            let result = GGUFFileReader::new(cursor);
            black_box(result)
        })
    });
}

fn benchmark_metadata_operations(c: &mut Criterion) {
    let mut metadata = FormatMetadata::new();

    // Pre-populate with some data
    for i in 0..1000 {
        metadata.insert(format!("key_{}", i), MetadataValue::U32(i));
    }

    c.bench_function("metadata_lookup", |b| {
        b.iter(|| {
            let key = format!("key_{}", black_box(500));
            black_box(metadata.get(&key))
        })
    });

    c.bench_function("metadata_iteration", |b| {
        b.iter(|| {
            for (key, value) in metadata.iter() {
                black_box((key, value));
            }
        })
    });
}

fn benchmark_tensor_operations(c: &mut Criterion) {
    let data = vec![0u8; 1024 * 1024]; // 1MB of data
    let tensor_data = TensorData::new_owned(data);
    let tensor_shape = gguf::tensor::TensorShape::new(vec![256, 256, 4]).unwrap();
    let tensor = gguf::tensor::TensorInfo::new(
        "benchmark_tensor".to_string(),
        tensor_shape,
        TensorType::F32,
        0,
    );

    c.bench_function("tensor_element_count", |b| b.iter(|| black_box(tensor.element_count())));

    c.bench_function("tensor_expected_size", |b| b.iter(|| black_box(tensor.expected_data_size())));

    c.bench_function("tensor_data_access", |b| b.iter(|| black_box(tensor_data.len())));
}

criterion_group!(
    benches,
    benchmark_gguf_read,
    benchmark_metadata_operations,
    benchmark_tensor_operations
);
criterion_main!(benches);
