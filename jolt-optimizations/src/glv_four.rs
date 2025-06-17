//! 4D GLV scalar multiplication implementations using Shamir's trick
//!
//! This module provides optimized scalar multiplication algorithms for BN254 G2 using
//! 4-dimensional decomposition with the Shamir trick and precomputed lookup tables.

use ark_bn254::{Fr, G2Projective};
// use ark_ec::Group;
use ark_ec::AdditiveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use rayon::prelude::*;

use crate::frobenius::frobenius_psi_power_projective;

/// Precomputed Shamir lookup table for 4-point scalar multiplication with signed combinations
/// Contains all 256 combinations: 16 point combinations × 16 sign patterns
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
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
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
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

/// Windowed lookup table processing 2 bits at a time (variable memory)
/// Uses 4-bit windows to reduce loop iterations
pub struct WindowedTable {
    pub window_tables: [G2Projective; 256], // 4^4 = 256 combinations for 2-bit windows
}

/// Signed windowed lookup table for efficient handling of positive/negative scalars
/// Contains all combinations with sign information baked in
pub struct SignedWindowedTable {
    pub signed_window_tables: [G2Projective; 4096], // 4^4 * 16 signs = 4096 combinations
}

/// 2-bit signed bases table - stores signed multiples for 2-bit processing (24x memory)
/// Contains: [P, 2P, 3P, -P, -2P, -3P] for each of the 4 bases
pub struct Windowed2SignedTable {
    pub signed_multiples: [G2Projective; 24], // 4 bases × 6 variants each (0,±1,±2,±3, but 0 is implicit)
}

/// 2-bit compact windowed table - 64 positive combinations (64x memory)  
/// Contains: positive 2-bit combinations with online sign handling
pub struct Windowed2CompactTable {
    pub positive_combinations: [G2Projective; 64], // 4^3 = 64 non-zero positive combinations
}

impl WindowedTable {
    /// Create windowed table for 2-bit processing
    pub fn new(bases: &[G2Projective; 4]) -> Self {
        let mut window_tables = [G2Projective::zero(); 256];

        // Precompute all 2-bit combinations
        window_tables
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, point)| {
                *point = G2Projective::zero();

                // Extract 2-bit coefficients for each base
                for i in 0..4 {
                    let coeff = (idx >> (i * 2)) & 0x3; // Extract 2 bits
                    match coeff {
                        0 => {},                                     // 0 * base
                        1 => *point += bases[i],                     // 1 * base
                        2 => *point += bases[i].double(),            // 2 * base
                        3 => *point += bases[i] + bases[i].double(), // 3 * base
                        _ => unreachable!(),
                    }
                }
            });

        Self { window_tables }
    }

    /// Get combination for windowed processing
    #[inline]
    pub fn get_window(&self, coeffs: &[u8; 4]) -> G2Projective {
        let mut idx = 0;
        for i in 0..4 {
            idx |= (coeffs[i] as usize & 0x3) << (i * 2);
        }
        self.window_tables[idx]
    }
}

impl SignedWindowedTable {
    /// Create signed windowed table for efficient sign handling
    pub fn new(bases: &[G2Projective; 4]) -> Self {
        let mut signed_window_tables = [G2Projective::zero(); 4096];

        // Precompute all combinations with all sign patterns
        signed_window_tables
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, point)| {
                *point = G2Projective::zero();

                // Lower 8 bits: 2-bit coefficients for each of 4 bases
                let coeff_mask = idx & 0xFF;
                // Upper 4 bits: sign for each base
                let sign_mask = (idx >> 8) & 0xF;

                // Extract 2-bit coefficients and signs for each base
                for i in 0..4 {
                    let coeff = (coeff_mask >> (i * 2)) & 0x3; // Extract 2 bits
                    let is_negative = (sign_mask >> i) & 1 == 1;

                    if coeff > 0 {
                        let base_contribution = match coeff {
                            1 => bases[i],                     // 1 * base
                            2 => bases[i].double(),            // 2 * base
                            3 => bases[i] + bases[i].double(), // 3 * base
                            _ => unreachable!(),
                        };

                        if is_negative {
                            *point -= base_contribution;
                        } else {
                            *point += base_contribution;
                        }
                    }
                }
            });

        Self {
            signed_window_tables,
        }
    }

    /// Get signed combination for windowed processing
    #[inline]
    pub fn get_signed_window(&self, coeffs: &[u8; 4], signs: &[bool; 4]) -> G2Projective {
        let mut coeff_idx = 0;
        let mut sign_idx = 0;

        for i in 0..4 {
            coeff_idx |= (coeffs[i] as usize & 0x3) << (i * 2);
            if signs[i] {
                sign_idx |= 1 << i;
            }
        }

        let idx = coeff_idx | (sign_idx << 8);
        self.signed_window_tables[idx]
    }
}

impl Windowed2SignedTable {
    /// Create 2-bit signed multiples table
    pub fn new(bases: &[G2Projective; 4]) -> Self {
        let mut signed_multiples = [G2Projective::zero(); 24];

        // For each base, store [1*base, 2*base, 3*base, -1*base, -2*base, -3*base]
        for (base_idx, &base) in bases.iter().enumerate() {
            let base_offset = base_idx * 6;
            signed_multiples[base_offset] = base; // 1*base
            signed_multiples[base_offset + 1] = base.double(); // 2*base
            signed_multiples[base_offset + 2] = base + base.double(); // 3*base
            signed_multiples[base_offset + 3] = -base; // -1*base
            signed_multiples[base_offset + 4] = -base.double(); // -2*base
            signed_multiples[base_offset + 5] = -(base + base.double()); // -3*base
        }

        Self { signed_multiples }
    }

    /// Get 2-bit windowed combination by building from signed multiples
    #[inline]
    pub fn get_windowed2(&self, coeffs: &[i8; 4]) -> G2Projective {
        let mut result = G2Projective::zero();

        for (i, &coeff) in coeffs.iter().enumerate() {
            if coeff != 0 {
                let base_offset = i * 6;
                let abs_coeff = coeff.abs() as usize;
                if abs_coeff <= 3 {
                    let entry = if coeff > 0 {
                        self.signed_multiples[base_offset + abs_coeff - 1] // Positive multiples
                    } else {
                        self.signed_multiples[base_offset + abs_coeff + 2] // Negative multiples
                    };
                    result += entry;
                }
            }
        }

        result
    }
}

impl Windowed2CompactTable {
    /// Create 2-bit compact table with positive combinations only
    pub fn new(bases: &[G2Projective; 4]) -> Self {
        let mut positive_combinations = [G2Projective::zero(); 64];

        // Generate all non-zero positive 2-bit combinations systematically
        let mut idx = 0;
        for combined in 1..=255u8 {
            // Skip 0 (all zeros)
            let c0 = combined & 0x3;
            let c1 = (combined >> 2) & 0x3;
            let c2 = (combined >> 4) & 0x3;
            let c3 = (combined >> 6) & 0x3;

            let mut combination = G2Projective::zero();
            if c0 > 0 {
                combination += match c0 {
                    1 => bases[0],
                    2 => bases[0].double(),
                    3 => bases[0] + bases[0].double(),
                    _ => unreachable!(),
                };
            }
            if c1 > 0 {
                combination += match c1 {
                    1 => bases[1],
                    2 => bases[1].double(),
                    3 => bases[1] + bases[1].double(),
                    _ => unreachable!(),
                };
            }
            if c2 > 0 {
                combination += match c2 {
                    1 => bases[2],
                    2 => bases[2].double(),
                    3 => bases[2] + bases[2].double(),
                    _ => unreachable!(),
                };
            }
            if c3 > 0 {
                combination += match c3 {
                    1 => bases[3],
                    2 => bases[3].double(),
                    3 => bases[3] + bases[3].double(),
                    _ => unreachable!(),
                };
            }

            positive_combinations[idx] = combination;
            idx += 1;
            if idx >= 64 {
                break;
            }
        }

        Self {
            positive_combinations,
        }
    }

    /// Get 2-bit combination with online sign handling
    #[inline]
    pub fn get_windowed2(&self, coeffs: &[i8; 4]) -> G2Projective {
        // Map signed coefficients to positive index and handle sign correction
        let mut positive_coeffs = [0u8; 4];
        let mut sign_correction = G2Projective::zero();

        for (i, &coeff) in coeffs.iter().enumerate() {
            positive_coeffs[i] = coeff.abs() as u8;
            if coeff < 0 {
                // Will need to subtract 2 * (positive contribution)
                let pos_contrib = match positive_coeffs[i] {
                    1 => self.positive_combinations[i],
                    2 => self.positive_combinations[i + 4],
                    3 => self.positive_combinations[i + 8],
                    _ => G2Projective::zero(),
                };
                sign_correction += pos_contrib.double();
            }
        }

        // Find positive combination (simplified lookup)
        let combined_idx = positive_coeffs[0]
            | (positive_coeffs[1] << 2)
            | (positive_coeffs[2] << 4)
            | (positive_coeffs[3] << 6);
        let result = if combined_idx > 0 && combined_idx < 64 {
            self.positive_combinations[combined_idx as usize - 1]
        } else {
            G2Projective::zero()
        };

        result - sign_correction
    }
}

pub struct WindowedData {
    pub windowed_tables: Vec<WindowedTable>,
}

impl WindowedData {
    pub fn new(points: &[G2Projective]) -> Self {
        let windowed_tables = points
            .par_iter()
            .map(|point| {
                let frobenius_bases = [
                    *point,
                    frobenius_psi_power_projective(point, 1),
                    frobenius_psi_power_projective(point, 2),
                    frobenius_psi_power_projective(point, 3),
                ];
                WindowedTable::new(&frobenius_bases)
            })
            .collect();

        Self { windowed_tables }
    }
}

/// Signed windowed data with 4096x memory usage per point (16x more than regular windowed)
pub struct SignedWindowedData {
    pub signed_windowed_tables: Vec<SignedWindowedTable>,
}

impl SignedWindowedData {
    pub fn new(points: &[G2Projective]) -> Self {
        let signed_windowed_tables = points
            .par_iter()
            .map(|point| {
                let frobenius_bases = [
                    *point,
                    frobenius_psi_power_projective(point, 1),
                    frobenius_psi_power_projective(point, 2),
                    frobenius_psi_power_projective(point, 3),
                ];
                SignedWindowedTable::new(&frobenius_bases)
            })
            .collect();

        Self {
            signed_windowed_tables,
        }
    }
}

/// 2-bit signed bases precomputed data with 24x memory usage per point
pub struct Windowed2SignedData {
    pub windowed2_signed_tables: Vec<Windowed2SignedTable>,
}

impl Windowed2SignedData {
    pub fn new(points: &[G2Projective]) -> Self {
        let windowed2_signed_tables = points
            .par_iter()
            .map(|point| {
                let frobenius_bases = [
                    *point,
                    frobenius_psi_power_projective(point, 1),
                    frobenius_psi_power_projective(point, 2),
                    frobenius_psi_power_projective(point, 3),
                ];
                Windowed2SignedTable::new(&frobenius_bases)
            })
            .collect();

        Self {
            windowed2_signed_tables,
        }
    }
}

/// 2-bit compact windowed precomputed data with 64x memory usage per point
pub struct Windowed2CompactData {
    pub windowed2_compact_tables: Vec<Windowed2CompactTable>,
}

impl Windowed2CompactData {
    pub fn new(points: &[G2Projective]) -> Self {
        let windowed2_compact_tables = points
            .par_iter()
            .map(|point| {
                let frobenius_bases = [
                    *point,
                    frobenius_psi_power_projective(point, 1),
                    frobenius_psi_power_projective(point, 2),
                    frobenius_psi_power_projective(point, 3),
                ];
                Windowed2CompactTable::new(&frobenius_bases)
            })
            .collect();

        Self {
            windowed2_compact_tables,
        }
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

/// Windowed precomputation function with 256x memory usage per point
///
/// Takes an array of G2 points and precomputes windowed tables for fast scalar multiplication:
/// - Frobenius endomorphism powers ψ(P), ψ²(P), ψ³(P) for each point P
/// - Windowed tables with 256 entries for 2-bit coefficient processing
///
/// This provides the same memory usage as full table but processes 2 bits at a time.
pub fn glv_four_precompute_windowed(points: &[G2Projective]) -> WindowedData {
    WindowedData::new(points)
}

/// 2-bit signed bases precomputation function with 24x memory usage per point
///
/// Takes an array of G2 points and precomputes 2-bit signed bases tables for fast scalar multiplication:
/// - Frobenius endomorphism powers ψ(P), ψ²(P), ψ³(P) for each point P
/// - Signed multiples tables with 24 entries: 4 bases × 6 variants each ([0,1,2,3]*base, [-1,-2,-3]*base)
///
/// This provides 90% memory savings vs glv_four_precompute() with 2-bit windowed processing.
pub fn glv_four_precompute_windowed2_signed(points: &[G2Projective]) -> Windowed2SignedData {
    Windowed2SignedData::new(points)
}

/// 2-bit compact windowed precomputation function with 64x memory usage per point
///
/// Takes an array of G2 points and precomputes 2-bit compact windowed tables for fast scalar multiplication:
/// - Frobenius endomorphism powers ψ(P), ψ²(P), ψ³(P) for each point P
/// - Compact windowed tables with 64 entries: 4^3 = 64 non-zero positive combinations
///
/// This provides 75% memory savings vs glv_four_precompute() with 2-bit windowed processing.
pub fn glv_four_precompute_windowed2_compact(points: &[G2Projective]) -> Windowed2CompactData {
    Windowed2CompactData::new(points)
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
            shamir_glv_mul_precomputed(
                shamir_table,
                &decomposed_scalar.k_bigint,
                &decomposed_scalar.signs,
            )
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

/// Windowed 4D GLV scalar multiplication using precomputed data and decomposed scalar
///
/// Uses windowed tables with 256x memory usage per point. Processes 2 bits at a time for reduced iterations.
///
/// # Arguments
/// * `windowed_data` - Windowed precomputed tables from `glv_four_precompute_windowed`
/// * `decomposed_scalar` - Already decomposed scalar from `DecomposedScalar::from_scalar`
///
/// # Returns
/// Vector of scalar multiplication results: [scalar * point[0], scalar * point[1], ...]
pub fn glv_four_scalar_mul_windowed_decomposed(
    windowed_data: &WindowedData,
    decomposed_scalar: &DecomposedScalar,
) -> Vec<G2Projective> {
    windowed_data
        .windowed_tables
        .par_iter()
        .map(|windowed_table| {
            shamir_glv_mul_windowed(
                windowed_table,
                &decomposed_scalar.k_bigint,
                &decomposed_scalar.signs,
            )
        })
        .collect()
}

/// Convenient wrapper for windowed 4D GLV scalar multiplication with automatic decomposition
///
/// Uses windowed tables with same memory as full table but 2-bit processing for faster computation.
///
/// # Arguments
/// * `windowed_data` - Windowed precomputed tables from `glv_four_precompute_windowed`
/// * `scalar` - The scalar to multiply with all points
///
/// # Returns
/// Vector of scalar multiplication results: [scalar * point[0], scalar * point[1], ...]
pub fn glv_four_scalar_mul_windowed(windowed_data: &WindowedData, scalar: Fr) -> Vec<G2Projective> {
    let decomposed_scalar = DecomposedScalar::from_scalar(scalar);
    glv_four_scalar_mul_windowed_decomposed(windowed_data, &decomposed_scalar)
}

/// 2-bit signed 4D GLV scalar multiplication using precomputed data and decomposed scalar
///
/// Uses 2-bit signed tables with 24x memory usage per point. Processes 2 bits at a time with signed multiples.
///
/// # Arguments
/// * `windowed2_signed_data` - 2-bit signed precomputed tables from `glv_four_precompute_windowed2_signed`
/// * `decomposed_scalar` - Already decomposed scalar from `DecomposedScalar::from_scalar`
///
/// # Returns
/// Vector of scalar multiplication results: [scalar * point[0], scalar * point[1], ...]
pub fn glv_four_scalar_mul_windowed2_signed_decomposed(
    windowed2_signed_data: &Windowed2SignedData,
    decomposed_scalar: &DecomposedScalar,
) -> Vec<G2Projective> {
    windowed2_signed_data
        .windowed2_signed_tables
        .par_iter()
        .map(|windowed2_signed_table| {
            shamir_glv_mul_windowed2_signed(
                windowed2_signed_table,
                &decomposed_scalar.k_bigint,
                &decomposed_scalar.signs,
            )
        })
        .collect()
}

/// Convenient wrapper for 2-bit signed 4D GLV scalar multiplication with automatic decomposition
///
/// Uses 2-bit signed tables with 90% memory savings compared to the full precomputed version.
///
/// # Arguments
/// * `windowed2_signed_data` - 2-bit signed precomputed tables from `glv_four_precompute_windowed2_signed`
/// * `scalar` - The scalar to multiply with all points
///
/// # Returns
/// Vector of scalar multiplication results: [scalar * point[0], scalar * point[1], ...]
pub fn glv_four_scalar_mul_windowed2_signed(
    windowed2_signed_data: &Windowed2SignedData,
    scalar: Fr,
) -> Vec<G2Projective> {
    let decomposed_scalar = DecomposedScalar::from_scalar(scalar);
    glv_four_scalar_mul_windowed2_signed_decomposed(windowed2_signed_data, &decomposed_scalar)
}

/// 2-bit compact 4D GLV scalar multiplication using precomputed data and decomposed scalar
///
/// Uses 2-bit compact tables with 64x memory usage per point. Processes 2 bits at a time with positive combinations.
///
/// # Arguments
/// * `windowed2_compact_data` - 2-bit compact precomputed tables from `glv_four_precompute_windowed2_compact`
/// * `decomposed_scalar` - Already decomposed scalar from `DecomposedScalar::from_scalar`
///
/// # Returns
/// Vector of scalar multiplication results: [scalar * point[0], scalar * point[1], ...]
// pub fn glv_four_scalar_mul_windowed2_compact_decomposed(
//     windowed2_compact_data: &Windowed2CompactData,
//     decomposed_scalar: &DecomposedScalar,
// ) -> Vec<G2Projective> {
//     windowed2_compact_data
//         .windowed2_compact_tables
//         .par_iter()
//         .map(|windowed2_compact_table| {
//             shamir_glv_mul_windowed2_compact(
//                 windowed2_compact_table,
//                 &decomposed_scalar.k_bigint,
//                 &decomposed_scalar.signs,
//             )
//         })
//         .collect()
// }

/// Convenient wrapper for 2-bit compact 4D GLV scalar multiplication with automatic decomposition
///
/// Uses 2-bit compact tables with 75% memory savings compared to the full precomputed version.
///
/// # Arguments
/// * `windowed2_compact_data` - 2-bit compact precomputed tables from `glv_four_precompute_windowed2_compact`
/// * `scalar` - The scalar to multiply with all points
///
/// # Returns
/// Vector of scalar multiplication results: [scalar * point[0], scalar * point[1], ...]
// pub fn glv_four_scalar_mul_windowed2_compact(
//     windowed2_compact_data: &Windowed2CompactData,
//     scalar: Fr,
// ) -> Vec<G2Projective> {
//     let decomposed_scalar = DecomposedScalar::from_scalar(scalar);
//     glv_four_scalar_mul_windowed2_compact_decomposed(windowed2_compact_data, &decomposed_scalar)
// }

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
pub fn glv_four_scalar_mul_online(scalar: Fr, points: &[G2Projective]) -> Vec<G2Projective> {
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
            shamir_glv_mul(
                &frobenius_powers,
                &decomposed_scalar.k_bigint,
                &decomposed_scalar.signs,
            )
        })
        .collect()
}

/// Precomputed data for fixed-base vector MSM in G2
/// 
/// This structure holds precomputed Frobenius endomorphism bases and Shamir tables
/// for a fixed base point, allowing efficient multiplication by multiple scalars.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct FixedBasePrecomputedG2 {
    /// The Frobenius endomorphism bases [P, ψ(P), ψ²(P), ψ³(P)]
    pub frobenius_bases: [G2Projective; 4],
    /// Precomputed Shamir table for all combinations and signs
    pub shamir_table: PrecomputedShamirTable,
}

impl FixedBasePrecomputedG2 {
    /// Create precomputed data for a fixed base point
    pub fn new(base: &G2Projective) -> Self {
        let frobenius_bases = [
            *base,
            frobenius_psi_power_projective(base, 1),
            frobenius_psi_power_projective(base, 2),
            frobenius_psi_power_projective(base, 3),
        ];
        let shamir_table = PrecomputedShamirTable::new(&frobenius_bases);
        
        Self {
            frobenius_bases,
            shamir_table,
        }
    }
    
    /// Multiply the fixed base by a single scalar using decomposed form
    pub fn mul_scalar_decomposed(&self, decomposed_scalar: &DecomposedScalar) -> G2Projective {
        shamir_glv_mul_precomputed(&self.shamir_table, &decomposed_scalar.k_bigint, &decomposed_scalar.signs)
    }
    
    /// Multiply the fixed base by a single scalar
    pub fn mul_scalar(&self, scalar: Fr) -> G2Projective {
        let decomposed_scalar = DecomposedScalar::from_scalar(scalar);
        self.mul_scalar_decomposed(&decomposed_scalar)
    }
    
    /// Multiply the fixed base by multiple scalars (all decomposed)
    pub fn mul_scalars_decomposed(&self, decomposed_scalars: &[DecomposedScalar]) -> Vec<G2Projective> {
        decomposed_scalars
            .par_iter()
            .map(|decomposed_scalar| self.mul_scalar_decomposed(decomposed_scalar))
            .collect()
    }
    
    /// Multiply the fixed base by multiple scalars
    pub fn mul_scalars(&self, scalars: &[Fr]) -> Vec<G2Projective> {
        scalars
            .par_iter()
            .map(|scalar| self.mul_scalar(*scalar))
            .collect()
    }
}

/// Fixed-base vector MSM for G2: multiply a single base point by multiple scalars
/// 
/// This function efficiently computes `base * scalars[i]` for all i using 4D GLV decomposition.
/// It precomputes the Frobenius endomorphism bases for the fixed base once and reuses them
/// for all scalar multiplications, providing significant speedup compared to naive approaches.
///
/// # Arguments
/// * `base` - The fixed G2 base point to multiply
/// * `scalars` - Vector of scalars to multiply the base by
///
/// # Returns
/// Vector of results where `result[i] = base * scalars[i]`
///
/// # Performance
/// This is optimal when you have a fixed base point and multiple different scalars,
/// as it avoids recomputing Frobenius endomorphisms for each scalar multiplication.
pub fn fixed_base_vector_msm_g2(base: &G2Projective, scalars: &[Fr]) -> Vec<G2Projective> {
    let precomputed = FixedBasePrecomputedG2::new(base);
    precomputed.mul_scalars(scalars)
}

#[cfg(test)]
mod fixed_base_tests {
    use super::*;
    use ark_bn254::G2Affine;
    use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_fixed_base_vector_msm_g2_correctness() {
        let mut rng = test_rng();

        // Generate a fixed base point and multiple scalars
        let base = G2Affine::rand(&mut rng).into_group();
        let scalars: Vec<Fr> = (0..10).map(|_| Fr::rand(&mut rng)).collect();

        // Compute using our optimized fixed-base MSM
        let results_optimized = fixed_base_vector_msm_g2(&base, &scalars);

        // Compute using naive approach for comparison
        let results_naive: Vec<G2Projective> = scalars
            .iter()
            .map(|scalar| base.mul_bigint(scalar.into_bigint()))
            .collect();

        // Verify results match
        for (i, (optimized, naive)) in results_optimized.iter().zip(results_naive.iter()).enumerate() {
            assert_eq!(
                optimized.into_affine(),
                naive.into_affine(),
                "Mismatch at index {}", i
            );
        }
    }
    
    #[test]
    fn test_fixed_base_precomputed_g2() {
        let mut rng = test_rng();
        let base = G2Affine::rand(&mut rng).into_group();
        
        // Test precomputed interface
        let precomputed = FixedBasePrecomputedG2::new(&base);
        
        // Test single scalar
        let scalar = Fr::rand(&mut rng);
        let result = precomputed.mul_scalar(scalar);
        let expected = base.mul_bigint(scalar.into_bigint());
        assert_eq!(result.into_affine(), expected.into_affine());
        
        // Test decomposed scalar
        let decomposed = DecomposedScalar::from_scalar(scalar);
        let result_decomposed = precomputed.mul_scalar_decomposed(&decomposed);
        assert_eq!(result_decomposed.into_affine(), expected.into_affine());
        
        // Test multiple scalars
        let scalars: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();
        let results = precomputed.mul_scalars(&scalars);
        for (i, (result, scalar)) in results.iter().zip(scalars.iter()).enumerate() {
            let expected = base.mul_bigint(scalar.into_bigint());
            assert_eq!(result.into_affine(), expected.into_affine(), "Mismatch at index {}", i);
        }
        
        // Test decomposed scalars
        let decomposed_scalars: Vec<DecomposedScalar> = scalars
            .iter()
            .map(|s| DecomposedScalar::from_scalar(*s))
            .collect();
        let results_decomposed = precomputed.mul_scalars_decomposed(&decomposed_scalars);
        for (i, (result, expected)) in results_decomposed.iter().zip(results.iter()).enumerate() {
            assert_eq!(result.into_affine(), expected.into_affine(), "Decomposed mismatch at index {}", i);
        }
    }

    #[test] 
    fn test_fixed_base_vector_msm_g2_edge_cases() {
        let mut rng = test_rng();
        let base = G2Affine::rand(&mut rng).into_group();

        // Test with single scalar
        let single_scalar = vec![Fr::rand(&mut rng)];
        let single_result = fixed_base_vector_msm_g2(&base, &single_scalar);
        let expected = base.mul_bigint(single_scalar[0].into_bigint());
        assert_eq!(single_result[0].into_affine(), expected.into_affine());

        // Test with zero scalar
        let zero_scalar = vec![Fr::from(0u64)];
        let zero_result = fixed_base_vector_msm_g2(&base, &zero_scalar);
        assert_eq!(zero_result[0], G2Projective::zero());

        // Test with empty vector
        let empty_scalars: Vec<Fr> = vec![];
        let empty_result = fixed_base_vector_msm_g2(&base, &empty_scalars);
        assert!(empty_result.is_empty());
    }
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

/// Windowed Shamir trick using WindowedTable (256x memory)
/// Note: This properly handles signed scalars by decomposing into positive/negative contributions
pub fn shamir_glv_mul_windowed(
    windowed_table: &WindowedTable,
    scalars: &[<Fr as PrimeField>::BigInt],
    signs: &[bool],
) -> G2Projective {
    assert_eq!(scalars.len(), 4);
    assert_eq!(signs.len(), 4);

    // Convert scalars to unsigned coefficients for windowed processing
    let scalar_coeffs: Vec<Vec<u8>> = scalars
        .par_iter()
        .map(|scalar| {
            let mut coeffs = Vec::new();
            let scalar_ref = scalar.as_ref();

            // Convert to base-4 coefficients (2 bits at a time)
            for limb in scalar_ref {
                for window_idx in 0..32 {
                    // 64 bits / 2 = 32 windows per limb
                    let window_bits = (limb >> (window_idx * 2)) & 0x3;
                    coeffs.push(window_bits as u8);
                }
            }

            coeffs
        })
        .collect();

    // Find maximum coefficient length
    let max_coeffs = scalar_coeffs
        .iter()
        .map(|coeffs| {
            coeffs
                .iter()
                .rposition(|&c| c != 0)
                .map(|pos| pos + 1)
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0);

    // Windowed processing: process 2 bits at a time
    let mut result = G2Projective::zero();

    for coeff_idx in (0..max_coeffs).rev() {
        // Square twice (process 2 bits)
        result = result.double().double();

        // Extract coefficients for this window
        let mut window_coeffs = [0u8; 4];
        for i in 0..4 {
            if coeff_idx < scalar_coeffs[i].len() {
                window_coeffs[i] = scalar_coeffs[i][coeff_idx];
            }
        }

        // Use windowed lookup if any coefficient is non-zero
        if window_coeffs.iter().any(|&c| c != 0) {
            // For efficiency, compute all positive and negative base contributions in one pass
            let mut positive_coeffs = [0u8; 4];
            let mut negative_coeffs = [0u8; 4];

            for i in 0..4 {
                if window_coeffs[i] != 0 {
                    if signs[i] {
                        negative_coeffs[i] = window_coeffs[i];
                    } else {
                        positive_coeffs[i] = window_coeffs[i];
                    }
                }
            }

            // Add positive contributions
            if positive_coeffs.iter().any(|&c| c != 0) {
                let positive_contribution = windowed_table.get_window(&positive_coeffs);
                result += positive_contribution;
            }

            // Subtract negative contributions
            if negative_coeffs.iter().any(|&c| c != 0) {
                let negative_contribution = windowed_table.get_window(&negative_coeffs);
                result -= negative_contribution;
            }
        }
    }

    result
}

/// 2-bit signed windowed Shamir trick using Windowed2SignedTable (24x memory)
pub fn shamir_glv_mul_windowed2_signed(
    windowed2_signed_table: &Windowed2SignedTable,
    scalars: &[<Fr as PrimeField>::BigInt],
    signs: &[bool],
) -> G2Projective {
    assert_eq!(scalars.len(), 4);
    assert_eq!(signs.len(), 4);

    // Convert scalars to signed coefficients for 2-bit windowed processing
    let scalar_coeffs: Vec<Vec<i8>> = scalars
        .par_iter()
        .zip(signs.par_iter())
        .map(|(scalar, &is_negative)| {
            let mut coeffs = Vec::new();
            let scalar_ref = scalar.as_ref();

            // Convert to base-4 coefficients (2 bits at a time)
            for limb in scalar_ref {
                for window_idx in 0..32 {
                    // 64 bits / 2 = 32 windows per limb
                    let window_bits = (limb >> (window_idx * 2)) & 0x3;
                    let signed_coeff = if is_negative {
                        -(window_bits as i8)
                    } else {
                        window_bits as i8
                    };
                    coeffs.push(signed_coeff);
                }
            }

            coeffs
        })
        .collect();

    // Find maximum coefficient length
    let max_coeffs = scalar_coeffs
        .iter()
        .map(|coeffs| {
            coeffs
                .iter()
                .rposition(|&c| c != 0)
                .map(|pos| pos + 1)
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0);

    // 2-bit windowed processing: process 2 bits at a time
    let mut result = G2Projective::zero();

    for coeff_idx in (0..max_coeffs).rev() {
        // Quadruple (process 2 bits): result = 4 * result
        result = result.double().double();

        // Extract coefficients for this window and use signed table lookup
        let mut window_coeffs = [0i8; 4];
        for i in 0..4 {
            if coeff_idx < scalar_coeffs[i].len() {
                window_coeffs[i] = scalar_coeffs[i][coeff_idx];
            }
        }

        // Use signed lookup if any coefficient is non-zero
        if window_coeffs.iter().any(|&c| c != 0) {
            let contribution = windowed2_signed_table.get_windowed2(&window_coeffs);
            result += contribution;
        }
    }

    result
}
