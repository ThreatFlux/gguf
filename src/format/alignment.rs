//! Data alignment utilities for GGUF files

use crate::format::constants::GGUF_DEFAULT_ALIGNMENT;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(not(feature = "std"))]
use core::{fmt, mem};

/// Calculate the padding needed to align to the specified boundary
pub fn calculate_padding(current_position: usize, alignment: usize) -> usize {
    if alignment == 0 || alignment == 1 {
        return 0;
    }

    let remainder = current_position % alignment;
    if remainder == 0 {
        0
    } else {
        alignment - remainder
    }
}

/// Align a position to the specified boundary
pub fn align_to(position: usize, alignment: usize) -> usize {
    position + calculate_padding(position, alignment)
}

/// Align a position to the default GGUF alignment boundary
pub fn align_to_default(position: usize) -> usize {
    align_to(position, GGUF_DEFAULT_ALIGNMENT)
}

/// Calculate the padding needed for default alignment
pub fn calculate_default_padding(position: usize) -> usize {
    calculate_padding(position, GGUF_DEFAULT_ALIGNMENT)
}

/// Check if a position is aligned to the specified boundary
pub fn is_aligned(position: usize, alignment: usize) -> bool {
    if alignment == 0 || alignment == 1 {
        return true;
    }
    position % alignment == 0
}

/// Check if a position is aligned to the default boundary
pub fn is_aligned_default(position: usize) -> bool {
    is_aligned(position, GGUF_DEFAULT_ALIGNMENT)
}

/// Utility to create padding bytes (zeros)
pub fn create_padding(size: usize) -> Vec<u8> {
    vec![0u8; size]
}

/// Calculate the aligned size for a given unaligned size
pub fn aligned_size(size: usize, alignment: usize) -> usize {
    if alignment == 0 || alignment == 1 {
        return size;
    }

    let remainder = size % alignment;
    if remainder == 0 {
        size
    } else {
        size + (alignment - remainder)
    }
}

/// Calculate the aligned size using default alignment
pub fn aligned_size_default(size: usize) -> usize {
    aligned_size(size, GGUF_DEFAULT_ALIGNMENT)
}

/// Check if an alignment value is valid (power of 2)
pub fn is_valid_alignment(alignment: usize) -> bool {
    alignment > 0 && (alignment & (alignment - 1)) == 0
}

/// Get the next power of 2 greater than or equal to the given value
pub fn next_power_of_2(mut n: usize) -> usize {
    if n == 0 {
        return 1;
    }

    if n & (n - 1) == 0 {
        return n; // Already a power of 2
    }

    // Find the next power of 2
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    #[cfg(feature = "std")]
    {
        if std::mem::size_of::<usize>() > 4 {
            n |= n >> 32;
        }
    }
    #[cfg(not(feature = "std"))]
    {
        if mem::size_of::<usize>() > 4 {
            n |= n >> 32;
        }
    }
    n + 1
}

/// Alignment information for a data section
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignmentInfo {
    /// Current position in the file/stream
    pub position: usize,
    /// Required alignment boundary
    pub alignment: usize,
    /// Calculated padding needed
    pub padding: usize,
    /// Final aligned position
    pub aligned_position: usize,
}

impl AlignmentInfo {
    /// Create alignment information for a given position and alignment
    pub fn new(position: usize, alignment: usize) -> Self {
        let padding = calculate_padding(position, alignment);
        let aligned_position = position + padding;

        Self { position, alignment, padding, aligned_position }
    }

    /// Create alignment information using default alignment
    pub fn new_default(position: usize) -> Self {
        Self::new(position, GGUF_DEFAULT_ALIGNMENT)
    }

    /// Check if padding is needed
    pub fn needs_padding(&self) -> bool {
        self.padding > 0
    }

    /// Get the padding as a vector of zero bytes
    pub fn padding_bytes(&self) -> Vec<u8> {
        create_padding(self.padding)
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for AlignmentInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AlignmentInfo {{ pos: {}, align: {}, padding: {}, aligned_pos: {} }}",
            self.position, self.alignment, self.padding, self.aligned_position
        )
    }
}

#[cfg(not(feature = "std"))]
impl fmt::Display for AlignmentInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AlignmentInfo {{ pos: {}, align: {}, padding: {}, aligned_pos: {} }}",
            self.position, self.alignment, self.padding, self.aligned_position
        )
    }
}

/// Helper struct for tracking alignment during file writing
#[derive(Debug, Clone)]
pub struct AlignmentTracker {
    /// Current position in the output stream
    pub position: usize,
    /// Default alignment to use
    pub default_alignment: usize,
}

impl AlignmentTracker {
    /// Create a new alignment tracker
    pub fn new(default_alignment: usize) -> Self {
        Self { position: 0, default_alignment }
    }

    /// Create a new alignment tracker with GGUF default alignment
    pub fn new_default() -> Self {
        Self::new(GGUF_DEFAULT_ALIGNMENT)
    }

    /// Advance the position by the given amount
    pub fn advance(&mut self, bytes: usize) {
        self.position += bytes;
    }

    /// Calculate padding needed for default alignment
    pub fn calculate_padding(&self) -> usize {
        calculate_padding(self.position, self.default_alignment)
    }

    /// Calculate padding needed for specific alignment
    pub fn calculate_padding_for(&self, alignment: usize) -> usize {
        calculate_padding(self.position, alignment)
    }

    /// Align to default boundary and return padding info
    pub fn align_default(&mut self) -> AlignmentInfo {
        let info = AlignmentInfo::new(self.position, self.default_alignment);
        self.position = info.aligned_position;
        info
    }

    /// Align to specific boundary and return padding info
    pub fn align_to(&mut self, alignment: usize) -> AlignmentInfo {
        let info = AlignmentInfo::new(self.position, alignment);
        self.position = info.aligned_position;
        info
    }

    /// Get current alignment info without advancing
    pub fn current_alignment_info(&self) -> AlignmentInfo {
        AlignmentInfo::new(self.position, self.default_alignment)
    }

    /// Check if currently aligned to default boundary
    pub fn is_aligned(&self) -> bool {
        is_aligned(self.position, self.default_alignment)
    }

    /// Check if currently aligned to specific boundary
    pub fn is_aligned_to(&self, alignment: usize) -> bool {
        is_aligned(self.position, alignment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_padding() {
        assert_eq!(calculate_padding(0, 32), 0);
        assert_eq!(calculate_padding(1, 32), 31);
        assert_eq!(calculate_padding(16, 32), 16);
        assert_eq!(calculate_padding(32, 32), 0);
        assert_eq!(calculate_padding(33, 32), 31);

        // Edge cases
        assert_eq!(calculate_padding(10, 1), 0);
        assert_eq!(calculate_padding(10, 0), 0);
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(0, 32), 0);
        assert_eq!(align_to(1, 32), 32);
        assert_eq!(align_to(16, 32), 32);
        assert_eq!(align_to(32, 32), 32);
        assert_eq!(align_to(33, 32), 64);
    }

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(0, 32));
        assert!(is_aligned(32, 32));
        assert!(is_aligned(64, 32));
        assert!(!is_aligned(1, 32));
        assert!(!is_aligned(33, 32));

        // Edge cases
        assert!(is_aligned(42, 1));
        assert!(is_aligned(42, 0));
    }

    #[test]
    fn test_aligned_size() {
        assert_eq!(aligned_size(0, 32), 0);
        assert_eq!(aligned_size(1, 32), 32);
        assert_eq!(aligned_size(16, 32), 32);
        assert_eq!(aligned_size(32, 32), 32);
        assert_eq!(aligned_size(33, 32), 64);
    }

    #[test]
    fn test_is_valid_alignment() {
        assert!(is_valid_alignment(1));
        assert!(is_valid_alignment(2));
        assert!(is_valid_alignment(4));
        assert!(is_valid_alignment(8));
        assert!(is_valid_alignment(16));
        assert!(is_valid_alignment(32));

        assert!(!is_valid_alignment(0));
        assert!(!is_valid_alignment(3));
        assert!(!is_valid_alignment(5));
        assert!(!is_valid_alignment(6));
        assert!(!is_valid_alignment(12));
    }

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(4), 4);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(15), 16);
        assert_eq!(next_power_of_2(16), 16);
        assert_eq!(next_power_of_2(17), 32);
    }

    #[test]
    fn test_alignment_info() {
        let info = AlignmentInfo::new(17, 32);
        assert_eq!(info.position, 17);
        assert_eq!(info.alignment, 32);
        assert_eq!(info.padding, 15);
        assert_eq!(info.aligned_position, 32);
        assert!(info.needs_padding());

        let info_aligned = AlignmentInfo::new(32, 32);
        assert_eq!(info_aligned.padding, 0);
        assert!(!info_aligned.needs_padding());
    }

    #[test]
    fn test_alignment_tracker() {
        let mut tracker = AlignmentTracker::new_default();
        assert_eq!(tracker.position, 0);
        assert!(tracker.is_aligned());

        tracker.advance(17);
        assert_eq!(tracker.position, 17);
        assert!(!tracker.is_aligned());
        assert_eq!(tracker.calculate_padding(), 15);

        let info = tracker.align_default();
        assert_eq!(info.padding, 15);
        assert_eq!(tracker.position, 32);
        assert!(tracker.is_aligned());
    }

    #[test]
    fn test_default_alignment_functions() {
        assert_eq!(align_to_default(17), 32);
        assert_eq!(calculate_default_padding(17), 15);
        assert!(is_aligned_default(32));
        assert!(!is_aligned_default(17));
        assert_eq!(aligned_size_default(17), 32);
    }

    #[test]
    fn test_create_padding() {
        let padding = create_padding(5);
        assert_eq!(padding.len(), 5);
        assert_eq!(padding, vec![0u8; 5]);

        let empty_padding = create_padding(0);
        assert!(empty_padding.is_empty());
    }

    #[test]
    fn test_alignment_info_display() {
        let info = AlignmentInfo::new(17, 32);
        let display_str = format!("{}", info);
        assert!(display_str.contains("17"));
        assert!(display_str.contains("32"));
        assert!(display_str.contains("15"));
    }
}
