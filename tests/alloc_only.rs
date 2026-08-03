//! Public API coverage for the supported `no_std + alloc` configuration.
//!
//! The test harness uses `std`, but every crate API exercised here is available
//! when `gguf-rs-lib` is built with only its `alloc` feature.

use gguf_rs_lib::format::{GGUFTensorType, GGUFValueType};
use gguf_rs_lib::metadata::{Metadata, MetadataArray, MetadataValue};
use gguf_rs_lib::tensor::{MemoryLayout, QuantizationParams, TensorInfo, TensorShape, TensorType};

#[test]
fn canonical_metadata_reexport_builds_validated_arrays() {
    fn accepts_format_value(_: &gguf_rs_lib::format::metadata::MetadataValue) {}

    let tokens = MetadataArray::new(
        GGUFValueType::String,
        vec![
            MetadataValue::String("alpha".to_string()),
            MetadataValue::String("beta".to_string()),
        ],
    )
    .expect("homogeneous metadata array should be valid");
    let reexported_value = MetadataValue::Array(Box::new(tokens));
    accepts_format_value(&reexported_value);

    let mut metadata = Metadata::new();
    metadata.insert("tokenizer.ggml.tokens".to_string(), reexported_value);
    metadata.insert("general.alignment".to_string(), MetadataValue::U32(64));

    metadata.validate().expect("metadata should satisfy GGUF invariants");
    assert_eq!(metadata.tensor_alignment().unwrap(), 64);
    assert_eq!(metadata.len(), 2);

    let malformed = MetadataArray {
        element_type: GGUFValueType::U32,
        length: 2,
        values: vec![MetadataValue::U32(7)],
    };
    assert!(malformed.validate().is_err());
    assert!(MetadataArray::new(
        GGUFValueType::U32,
        vec![MetadataValue::String("wrong type".to_string())]
    )
    .is_err());
}

#[test]
fn shapes_and_layouts_follow_ggml_dimension_order() {
    let matrix = TensorShape::matrix(2, 3).expect("matrix shape should be valid");
    assert_eq!(matrix.dims(), &[3, 2]);
    assert_eq!(matrix.calculate_strides(), vec![1, 3]);

    let broadcast = TensorShape::new(vec![4, 1, 8])
        .unwrap()
        .broadcast_with(&TensorShape::new(vec![1, 3]).unwrap())
        .expect("GGML-order shapes should broadcast from dimension zero");
    assert_eq!(broadcast.dims(), &[4, 3, 8]);

    let product = TensorShape::new(vec![3, 2])
        .unwrap()
        .matmul_output_shape(&TensorShape::new(vec![4, 3]).unwrap())
        .expect("[K, M] x [N, K] should be compatible");
    assert_eq!(product.dims(), &[4, 2]);

    let info = TensorInfo::new("weights".to_string(), matrix, TensorType::F32, 0);
    let layout = info.calculate_layout().expect("F32 layout should be representable");
    assert_eq!(layout.memory_layout, MemoryLayout::Ggml);
    assert_eq!(layout.strides, vec![4, 12]);
    assert_eq!(layout.alignment, 1);
    assert_eq!(info.checked_expected_data_size().unwrap(), 24);
}

#[test]
fn current_tensor_type_ids_have_canonical_block_geometry() {
    let geometries = [
        (34, GGUFTensorType::TQ1_0, 256, 54),
        (35, GGUFTensorType::TQ2_0, 256, 66),
        (39, GGUFTensorType::MXFP4, 32, 17),
        (40, GGUFTensorType::NVFP4, 64, 36),
        (41, GGUFTensorType::Q1_0, 128, 18),
        (42, GGUFTensorType::Q2_0, 64, 18),
    ];

    for (raw_id, tensor_type, block_elements, block_bytes) in geometries {
        assert_eq!(GGUFTensorType::from_u32(raw_id).unwrap(), tensor_type);
        assert_eq!(tensor_type.block_size(), block_elements);
        assert_eq!(tensor_type.block_size_bytes(), Some(block_bytes));
        assert_eq!(
            tensor_type.checked_calculate_size(block_elements as u64),
            Some(block_bytes as u64)
        );

        let params = QuantizationParams::for_type(tensor_type);
        assert!(params.is_supported());
        assert_eq!(params.block_size, block_elements);
        assert_eq!(params.block_size_bytes, block_bytes);
    }
}
