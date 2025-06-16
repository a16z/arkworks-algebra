//! Multi-scalar multiplication (MSM) implementations using 4D decomposition
//!
//! This module provides optimized MSM algorithms for BN254 G2 using
//! the Shamir trick with precomputed lookup tables.

use ark_bn254::{Fr, G2Projective};
use ark_ec::Group;
use ark_ff::PrimeField;
use ark_std::Zero;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;

use crate::frobenius::frobenius_psi_power_projective;

/// Precomputed Shamir lookup table for 4-point MSM with signed combinations
/// Contains all 256 combinations: 16 point combinations × 16 sign patterns
pub struct PrecomputedShamirTable {
    pub table: [G2Projective; 256], // 2^4 point combinations × 2^4 sign patterns
}

impl PrecomputedShamirTable {
    /// Create precomputed table for [P, ψ(P), ψ²(P), ψ³(P)] with all sign combinations
    pub fn new(bases: &[G2Projective; 4]) -> Self {
        let mut table = [G2Projective::zero(); 256];

        // Use parallelism to compute all combinations
        table.par_iter_mut().enumerate().for_each(|(idx, point)| {
            let point_mask = idx & 0xF; // Lower 4 bits: which points to include
            let sign_mask = idx >> 4; // Upper 4 bits: which points to negate

            *point = G2Projective::zero();
            for i in 0..4 {
                if (point_mask >> i) & 1 == 1 {
                    if (sign_mask >> i) & 1 == 1 {
                        *point -= bases[i]; // Negative contribution
                    } else {
                        *point += bases[i]; // Positive contribution
                    }
                }
            }
        });

        Self { table }
    }

    /// Get the precomputed point for given point mask and sign mask
    #[inline]
    pub fn get(&self, point_mask: usize, sign_mask: usize) -> G2Projective {
        self.table[point_mask | (sign_mask << 4)]
    }
}

/// Enhanced precomputed data combining Frobenius powers and Shamir table
pub struct PrecomputedShamirData {
    pub shamir_tables: Vec<PrecomputedShamirTable>,
}

impl PrecomputedShamirData {
    pub fn new(points: &[G2Projective]) -> Self {
        let shamir_tables = points
            .par_iter()
            .map(|point| {
                let frobenius_bases = [
                    *point,
                    frobenius_psi_power_projective(point, 1),
                    frobenius_psi_power_projective(point, 2),
                    frobenius_psi_power_projective(point, 3),
                ];
                PrecomputedShamirTable::new(&frobenius_bases)
            })
            .collect();

        Self { shamir_tables }
    }
}

/// Global profiling data for MSM operations
static MSM_PROFILE: LazyLock<MSMProfile> = LazyLock::new(|| MSMProfile::default());

/// Global profiling data for precomputed MSM operations
static MSM_PROFILE_PRECOMPUTED: LazyLock<MSMProfile> = LazyLock::new(|| MSMProfile::default());

/// Profiling data for MSM operations
#[derive(Default)]
pub struct MSMProfile {
    pub window_processing_ns: AtomicU64,
    pub bucket_assignment_ns: AtomicU64,
    pub bucket_summation_ns: AtomicU64,
    pub window_combination_ns: AtomicU64,
    pub total_calls: AtomicU64,
}

impl MSMProfile {
    pub fn print_stats(&self) {
        let calls = self.total_calls.load(Ordering::Relaxed);
        if calls == 0 {
            println!("  MSM Profile: No calls recorded");
            return;
        }

        let window_proc = self.window_processing_ns.load(Ordering::Relaxed) as f64 / calls as f64;
        let bucket_assign = self.bucket_assignment_ns.load(Ordering::Relaxed) as f64 / calls as f64;
        let bucket_sum = self.bucket_summation_ns.load(Ordering::Relaxed) as f64 / calls as f64;
        let window_comb = self.window_combination_ns.load(Ordering::Relaxed) as f64 / calls as f64;

        println!("    MSM Profile (avg per call over {} calls):", calls);
        println!("      Main loop:        {:.1} μs", window_proc / 1000.0);
        println!("      Bit extraction:   {:.1} μs", bucket_assign / 1000.0);
        println!("      Precomputation:   {:.1} μs", bucket_sum / 1000.0);
        println!("      Window combination: {:.1} μs", window_comb / 1000.0);
        println!(
            "      Total:            {:.1} μs",
            (window_proc + bucket_assign + bucket_sum + window_comb) / 1000.0
        );
    }
}

/// Shamir trick for 4-point multi-scalar multiplication with parallelism
pub fn msm_small_66bit(
    bases: &[G2Projective],
    scalars: &[<Fr as PrimeField>::BigInt],
    signs: &[bool],
) -> G2Projective {
    assert_eq!(bases.len(), 4);
    assert_eq!(scalars.len(), 4);
    assert_eq!(signs.len(), 4);

    let _start_total = std::time::Instant::now();

    // Convert scalars to bit representations with signs
    let start_bit_extraction = std::time::Instant::now();

    let bit_arrays: Vec<Vec<i8>> = scalars
        .par_iter()
        .zip(signs.par_iter())
        .map(|(scalar, &is_negative)| {
            let mut bits = Vec::new();
            let scalar_ref = scalar.as_ref();

            // Extract bits from all limbs
            for limb in scalar_ref {
                for bit_idx in 0..64 {
                    let bit = (limb >> bit_idx) & 1;
                    let signed_bit = if bit == 1 {
                        if is_negative {
                            -1
                        } else {
                            1
                        }
                    } else {
                        0
                    };
                    bits.push(signed_bit);
                }
            }

            bits
        })
        .collect();

    // Find maximum bit length across all scalars
    let max_bits = bit_arrays
        .iter()
        .map(|bits| {
            bits.iter()
                .rposition(|&b| b != 0)
                .map(|pos| pos + 1)
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0);

    let bit_extraction_time = start_bit_extraction.elapsed().as_nanos() as u64;

    // Precompute all combinations: P0, P1, P2, P3, P0+P1, P0+P2, ..., P0+P1+P2+P3
    let start_precompute = std::time::Instant::now();

    let mut precomputed = vec![G2Projective::zero(); 16]; // 2^4 combinations

    // Use parallelism to compute precomputed table
    precomputed
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, point)| {
            *point = G2Projective::zero();
            for i in 0..4 {
                if (idx >> i) & 1 == 1 {
                    *point += bases[i];
                }
            }
        });

    let precompute_time = start_precompute.elapsed().as_nanos() as u64;

    // Shamir trick: process bits from MSB to LSB
    let start_shamir = std::time::Instant::now();

    let mut result = G2Projective::zero();

    for bit_idx in (0..max_bits).rev() {
        // Double the accumulator
        result = result.double();

        // Determine which precomputed point to add based on current bits
        let mut lookup_idx = 0;
        for i in 0..4 {
            if bit_idx < bit_arrays[i].len() {
                let bit = bit_arrays[i][bit_idx];
                if bit == 1 {
                    lookup_idx |= 1 << i;
                } else if bit == -1 {
                    lookup_idx |= 1 << i;
                    // We'll handle negation below
                }
            }
        }

        if lookup_idx != 0 {
            let mut point_to_add = precomputed[lookup_idx];

            // Handle negation based on sign_mask
            // For each bit that should be negative, we need to subtract instead of add
            // This is equivalent to adding the negation of the corresponding base point
            let mut sign_mask = 0;
            for i in 0..4 {
                if bit_idx < bit_arrays[i].len()
                    && bit_arrays[i][bit_idx] == -1
                    && ((lookup_idx >> i) & 1) == 1
                {
                    sign_mask |= 1 << i;
                }
            }

            if sign_mask != 0 {
                // Compute the correction: subtract 2 * (sum of negative components)
                let mut correction = G2Projective::zero();
                for i in 0..4 {
                    if (sign_mask >> i) & 1 == 1 {
                        // Get the individual base point from the precomputed table
                        let base_lookup = 1 << i;
                        correction += precomputed[base_lookup];
                    }
                }
                point_to_add -= correction.double();
            }

            result += point_to_add;
        }
    }

    let shamir_time = start_shamir.elapsed().as_nanos() as u64;

    // Record timing
    MSM_PROFILE
        .bucket_assignment_ns
        .fetch_add(bit_extraction_time, Ordering::Relaxed);
    MSM_PROFILE
        .bucket_summation_ns
        .fetch_add(precompute_time, Ordering::Relaxed);
    MSM_PROFILE
        .window_processing_ns
        .fetch_add(shamir_time, Ordering::Relaxed);
    MSM_PROFILE
        .window_combination_ns
        .fetch_add(0, Ordering::Relaxed);
    MSM_PROFILE.total_calls.fetch_add(1, Ordering::Relaxed);

    result
}

/// Optimized Shamir trick using precomputed table (no online precomputation)
pub fn msm_small_66bit_precomputed(
    shamir_table: &PrecomputedShamirTable,
    scalars: &[<Fr as PrimeField>::BigInt],
    signs: &[bool],
) -> G2Projective {
    assert_eq!(scalars.len(), 4);
    assert_eq!(signs.len(), 4);

    let _start_total = std::time::Instant::now();

    // Convert scalars to bit representations with signs
    let start_bit_extraction = std::time::Instant::now();

    let bit_arrays: Vec<Vec<i8>> = scalars
        .par_iter()
        .zip(signs.par_iter())
        .map(|(scalar, &is_negative)| {
            let mut bits = Vec::new();
            let scalar_ref = scalar.as_ref();

            // Extract bits from all limbs
            for limb in scalar_ref {
                for bit_idx in 0..64 {
                    let bit = (limb >> bit_idx) & 1;
                    let signed_bit = if bit == 1 {
                        if is_negative {
                            -1
                        } else {
                            1
                        }
                    } else {
                        0
                    };
                    bits.push(signed_bit);
                }
            }

            bits
        })
        .collect();

    // Find maximum bit length across all scalars
    let max_bits = bit_arrays
        .iter()
        .map(|bits| {
            bits.iter()
                .rposition(|&b| b != 0)
                .map(|pos| pos + 1)
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0);

    let bit_extraction_time = start_bit_extraction.elapsed().as_nanos() as u64;

    // Shamir trick: process bits from MSB to LSB (using precomputed table)
    let start_shamir = std::time::Instant::now();

    let mut result = G2Projective::zero();

    for bit_idx in (0..max_bits).rev() {
        // Double the accumulator
        result = result.double();

        // Extract point mask and sign mask in a single loop
        let mut point_mask = 0;
        let mut sign_mask = 0;
        for i in 0..4 {
            if bit_idx < bit_arrays[i].len() {
                let bit = bit_arrays[i][bit_idx];
                if bit != 0 {
                    point_mask |= 1 << i;
                    if bit == -1 {
                        sign_mask |= 1 << i;
                    }
                }
            }
        }

        // Use optimized lookup with precomputed signed combinations
        if point_mask != 0 {
            result += shamir_table.get(point_mask, sign_mask);
        }
    }

    let shamir_time = start_shamir.elapsed().as_nanos() as u64;

    // Record timing (no precomputation time since it's done offline!)
    MSM_PROFILE_PRECOMPUTED
        .bucket_assignment_ns
        .fetch_add(bit_extraction_time, Ordering::Relaxed);
    MSM_PROFILE_PRECOMPUTED
        .bucket_summation_ns
        .fetch_add(0, Ordering::Relaxed); // No online precomputation!
    MSM_PROFILE_PRECOMPUTED
        .window_processing_ns
        .fetch_add(shamir_time, Ordering::Relaxed);
    MSM_PROFILE_PRECOMPUTED
        .window_combination_ns
        .fetch_add(0, Ordering::Relaxed);
    MSM_PROFILE_PRECOMPUTED
        .total_calls
        .fetch_add(1, Ordering::Relaxed);

    result
}

/// Print profiling statistics for MSM operations
pub fn print_msm_profile() {
    println!("=== MSM Profiling Statistics ===");
    println!("Standard MSM:");
    MSM_PROFILE.print_stats();
    println!("\nPrecomputed MSM:");
    MSM_PROFILE_PRECOMPUTED.print_stats();
}
