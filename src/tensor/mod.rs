//! Tensor descriptors, raw storage, shapes, and quantization geometry
//!
//! Recognized quantized formats expose exact raw block geometry, but this crate
//! does not quantize, dequantize, or execute tensors. See the format-support
//! guide for the current type registry and payload boundaries.

pub mod data;
pub mod info;
pub mod quantization;
pub mod shape;
pub mod tensor_type;

pub use data::*;
pub use info::*;
pub use quantization::*;
pub use shape::*;
pub use tensor_type::*;
