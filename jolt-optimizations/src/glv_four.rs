//! 4D GLV scalar multiplication implementations using Shamir's trick
//!
//! This module provides optimized scalar multiplication algorithms for BN254 G2 using
//! 4-dimensional decomposition with the Shamir trick and precomputed lookup tables.

use ark_bn254::{Fr, G2Projective};
use ark_ec::Group;
use ark_ff::PrimeField;
use ark_std::Zero;
use rayon::prelude::*;

use crate::frobenius::frobenius_psi_power_projective;

/// Precomputed Shamir lookup table for 4-point scalar multiplication with signed combinations
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

/// Unified precomputation function for 4D GLV scalar multiplication
/// 
/// Takes an array of G2 points and precomputes everything needed for fast scalar multiplication:
/// - Frobenius endomorphism powers ψ(P), ψ²(P), ψ³(P) for each point P
/// - Shamir lookup tables with all 256 signed combinations for each point
/// 
/// This is the main function users should call to precompute data for repeated scalar multiplications.
pub fn glv_four_precompute(points: &[G2Projective]) -> PrecomputedShamirData {
    PrecomputedShamirData::new(points)
}

/// Decomposed scalar representation for 4D GLV
#[derive(Clone, Debug)]
pub struct DecomposedScalar {
    pub k_bigint: [<Fr as PrimeField>::BigInt; 4],
    pub signs: [bool; 4],
}

impl DecomposedScalar {
    /// Decompose a scalar into 4D GLV form
    pub fn from_scalar(scalar: Fr) -> Self {
        use crate::decomposition::{decompose_scalar_table_based, fr_to_bigint, u128_to_fr};
        
        let scalar_bigint = fr_to_bigint(scalar);
        let (coeffs, signs) = decompose_scalar_table_based(&scalar_bigint);
        
        let k_bigint = [
            u128_to_fr(coeffs[0]).into_bigint(),
            u128_to_fr(coeffs[1]).into_bigint(),
            u128_to_fr(coeffs[2]).into_bigint(),
            u128_to_fr(coeffs[3]).into_bigint(),
        ];
        
        Self { k_bigint, signs }
    }
}

/// Core 4D GLV scalar multiplication using precomputed data and decomposed scalar
/// 
/// This is the main function that performs scalar multiplication using precomputed Shamir tables
/// and an already decomposed scalar. This is most efficient when you have a fixed scalar
/// that you want to multiply with many different points.
/// 
/// # Arguments
/// * `precomputed_data` - Precomputed Shamir tables from `glv_four_precompute`
/// * `decomposed_scalar` - Already decomposed scalar from `DecomposedScalar::from_scalar`
/// 
/// # Returns
/// Vector of scalar multiplication results: [scalar * point[0], scalar * point[1], ...]
pub fn glv_four_scalar_mul_decomposed(
    precomputed_data: &PrecomputedShamirData,
    decomposed_scalar: &DecomposedScalar,
) -> Vec<G2Projective> {
    precomputed_data
        .shamir_tables
        .par_iter()
        .map(|shamir_table| {
            shamir_glv_mul_precomputed(shamir_table, &decomposed_scalar.k_bigint, &decomposed_scalar.signs)
        })
        .collect()
}

/// Convenient wrapper for 4D GLV scalar multiplication with automatic decomposition
/// 
/// This function decomposes the scalar once and then performs scalar multiplication
/// for all points. Use this when you have a single scalar and multiple points.
/// For repeated operations with the same scalar, use `DecomposedScalar::from_scalar` 
/// and `glv_four_scalar_mul_decomposed` directly.
/// 
/// # Arguments
/// * `precomputed_data` - Precomputed Shamir tables from `glv_four_precompute`
/// * `scalar` - The scalar to multiply with all points
/// 
/// # Returns
/// Vector of scalar multiplication results: [scalar * point[0], scalar * point[1], ...]
pub fn glv_four_scalar_mul(
    precomputed_data: &PrecomputedShamirData,
    scalar: Fr,
) -> Vec<G2Projective> {
    let decomposed_scalar = DecomposedScalar::from_scalar(scalar);
    glv_four_scalar_mul_decomposed(precomputed_data, &decomposed_scalar)
}

/// 4D GLV scalar multiplication with online Frobenius computation
/// 
/// This function takes a scalar and array of points, decomposes the scalar once,
/// and then computes Frobenius powers online for each point before doing scalar multiplication.
/// This is useful when you have a fixed scalar but don't want to precompute Shamir tables.
/// 
/// # Arguments
/// * `scalar` - The scalar to multiply with all points
/// * `points` - Array of G2 points
/// 
/// # Returns
/// Vector of scalar multiplication results: [scalar * point[0], scalar * point[1], ...]
pub fn glv_four_scalar_mul_online(
    scalar: Fr,
    points: &[G2Projective],
) -> Vec<G2Projective> {
    let decomposed_scalar = DecomposedScalar::from_scalar(scalar);
    
    points
        .par_iter()
        .map(|point| {
            // Compute Frobenius powers on the fly for each point
            let frobenius_powers = [
                *point,
                frobenius_psi_power_projective(point, 1),
                frobenius_psi_power_projective(point, 2),
                frobenius_psi_power_projective(point, 3),
            ];
            shamir_glv_mul(&frobenius_powers, &decomposed_scalar.k_bigint, &decomposed_scalar.signs)
        })
        .collect()
}

/// Shamir trick for 4-point scalar multiplication with parallelism
pub fn shamir_glv_mul(
    bases: &[G2Projective],
    scalars: &[<Fr as PrimeField>::BigInt],
    signs: &[bool],
) -> G2Projective {
    assert_eq!(bases.len(), 4);
    assert_eq!(scalars.len(), 4);
    assert_eq!(signs.len(), 4);

    // Convert scalars to bit representations with signs
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

    // Precompute all combinations: P0, P1, P2, P3, P0+P1, P0+P2, ..., P0+P1+P2+P3
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

    // Shamir trick: process bits from MSB to LSB
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

    result
}

/// Optimized Shamir trick using precomputed table (no online precomputation)
pub fn shamir_glv_mul_precomputed(
    shamir_table: &PrecomputedShamirTable,
    scalars: &[<Fr as PrimeField>::BigInt],
    signs: &[bool],
) -> G2Projective {
    assert_eq!(scalars.len(), 4);
    assert_eq!(signs.len(), 4);

    // Convert scalars to bit representations with signs
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

    // Shamir trick: process bits from MSB to LSB (using precomputed table)
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

    result
}