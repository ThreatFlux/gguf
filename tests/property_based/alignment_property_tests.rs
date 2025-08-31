//! Property-based tests for alignment calculations

use gguf::format::{align_to, calculate_padding, is_aligned, GGUF_DEFAULT_ALIGNMENT};
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_alignment_properties(
        value in 0usize..100000,
        alignment_power in 0u32..10u32
    ) {
        let alignment = 1usize << alignment_power; // Generate powers of 2: 1, 2, 4, 8, ..., 512

        let aligned = align_to(value, alignment);

        // Aligned value should be >= original value
        prop_assert!(aligned >= value);

        // Aligned value should be aligned
        prop_assert!(is_aligned(aligned, alignment));

        // Aligned value should be the smallest such value
        if value > 0 && !is_aligned(value, alignment) {
            prop_assert!(aligned < value + alignment);
        }
    }

    #[test]
    fn test_padding_properties(
        value in 0usize..100000,
        alignment_power in 0u32..10u32
    ) {
        let alignment = 1usize << alignment_power;

        let padding = calculate_padding(value, alignment);

        // Padding should be non-negative and less than alignment
        prop_assert!(padding < alignment);

        // Value + padding should be aligned
        prop_assert!(is_aligned(value + padding, alignment));

        // If already aligned, padding should be 0
        if is_aligned(value, alignment) {
            prop_assert_eq!(padding, 0);
        }
    }

    #[test]
    fn test_alignment_consistency(
        value in 0usize..100000,
        alignment_power in 0u32..10u32
    ) {
        let alignment = 1usize << alignment_power;

        let aligned = align_to(value, alignment);
        let padding = calculate_padding(value, alignment);

        // align_to and calculate_padding should be consistent
        prop_assert_eq!(aligned, value + padding);
    }

    #[test]
    fn test_default_alignment_properties(
        value in 0usize..100000
    ) {
        let aligned = align_to(value, GGUF_DEFAULT_ALIGNMENT);
        let padding = calculate_padding(value, GGUF_DEFAULT_ALIGNMENT);

        // Should work with default alignment
        prop_assert!(is_aligned(aligned, GGUF_DEFAULT_ALIGNMENT));
        prop_assert_eq!(aligned, value + padding);
        prop_assert!(aligned >= value);
    }

    #[test]
    fn test_alignment_idempotency(
        value in 0usize..100000,
        alignment_power in 0u32..10u32
    ) {
        let alignment = 1usize << alignment_power;

        let aligned_once = align_to(value, alignment);
        let aligned_twice = align_to(aligned_once, alignment);

        // Aligning an already aligned value should not change it
        prop_assert_eq!(aligned_once, aligned_twice);

        // Padding an already aligned value should be 0
        prop_assert_eq!(calculate_padding(aligned_once, alignment), 0);
    }

    #[test]
    fn test_alignment_with_powers_of_two(
        value in 0usize..65536,
        power in 0u8..16 // 2^0 to 2^15
    ) {
        let alignment = 1usize << power; // 2^power

        let aligned = align_to(value, alignment);

        prop_assert!(is_aligned(aligned, alignment));
        prop_assert!(aligned >= value);
        prop_assert!(aligned < value + alignment);
    }
}

// Test specific edge cases
proptest! {
    #[test]
    fn test_zero_alignment_edge_case(
        value in 1usize..1000
    ) {
        // Testing with alignment 0 - the implementation treats 0 as no alignment needed
        // So it should return the original value unchanged
        let result = align_to(value, 0);
        prop_assert_eq!(result, value, "align_to with 0 alignment should return original value");

        // Padding should also be 0
        let padding = calculate_padding(value, 0);
        prop_assert_eq!(padding, 0, "calculate_padding with 0 alignment should return 0");
    }

    #[test]
    fn test_zero_value_alignment(
        alignment_power in 0u32..10u32
    ) {
        let alignment = 1usize << alignment_power;

        // Zero should always be aligned to any alignment
        prop_assert!(is_aligned(0, alignment));
        prop_assert_eq!(align_to(0, alignment), 0);
        prop_assert_eq!(calculate_padding(0, alignment), 0);
    }

    #[test]
    fn test_alignment_boundary_values(
        alignment_power in 1u32..10u32  // Start from 1 to get at least 2
    ) {
        let alignment = 1usize << alignment_power;

        // Test values around alignment boundaries
        let boundary = alignment;

        // Value exactly at boundary should be aligned
        prop_assert!(is_aligned(boundary, alignment));
        prop_assert_eq!(align_to(boundary, alignment), boundary);
        prop_assert_eq!(calculate_padding(boundary, alignment), 0);

        // Value one less than boundary should need padding
        if boundary > 0 {
            prop_assert!(!is_aligned(boundary - 1, alignment));
            prop_assert_eq!(align_to(boundary - 1, alignment), boundary);
            prop_assert_eq!(calculate_padding(boundary - 1, alignment), 1);
        }
    }
}
