use ark_bn254::{Fq, Fr, G1Projective};
use ark_ec::{AdditiveGroup, PrimeGroup};
use ark_ff::{BigInteger, MontFp, PrimeField};
use ark_std::ops::{AddAssign, Neg};
use ark_std::Zero;
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed};
use rayon::prelude::*;

use crate::decomp_2d::{decompose_scalar_2d, glv_endomorphism};
/// Precomputed Shamir lookup table for 2-point scalar multiplication with signed combinations
/// Contains all 16 combinations: 4 point combinations × 4 sign patterns
#[derive(Clone, Debug)]
pub struct PrecomputedShamir2Table {
    pub table: [G1Projective; 16], // 2^2 point combinations × 2^2 sign patterns
}

impl PrecomputedShamir2Table {
    /// Create precomputed table for [P, λ(P)] with all sign combinations
    pub fn new(bases: &[G1Projective; 2]) -> Self {
        let mut table = [G1Projective::zero(); 16];

        // Use parallelism to compute all combinations
        table.par_iter_mut().enumerate().for_each(|(idx, point)| {
            let point_mask = idx & 0x3; // Lower 2 bits: which points to include
            let sign_mask = idx >> 2; // Upper 2 bits: which points to negate

            *point = G1Projective::zero();
            for i in 0..2 {
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
    pub fn get(&self, point_mask: usize, sign_mask: usize) -> G1Projective {
        self.table[point_mask | (sign_mask << 2)]
    }
}

/// Enhanced precomputed data combining GLV endomorphism and Shamir table
pub struct PrecomputedShamir2Data {
    pub shamir_tables: Vec<PrecomputedShamir2Table>,
}

impl PrecomputedShamir2Data {
    pub fn new(points: &[G1Projective]) -> Self {
        let shamir_tables = points
            .par_iter()
            .map(|point| {
                let glv_bases = [*point, glv_endomorphism(point)];
                PrecomputedShamir2Table::new(&glv_bases)
            })
            .collect();

        Self { shamir_tables }
    }
}

/// 2-bit signed bases table for 2D GLV - stores signed multiples for 2-bit processing (12x memory)
/// Contains: [P, 2P, 3P, -P, -2P, -3P] for each of the 2 bases
pub struct Windowed2Signed2Table {
    pub signed_multiples: [G1Projective; 12], // 2 bases × 6 variants each
}

impl Windowed2Signed2Table {
    /// Create 2-bit signed multiples table for 2 bases
    pub fn new(bases: &[G1Projective; 2]) -> Self {
        let mut signed_multiples = [G1Projective::zero(); 12];

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
    pub fn get_windowed2(&self, coeffs: &[i8; 2]) -> G1Projective {
        let mut result = G1Projective::zero();

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

/// 2-bit signed bases precomputed data with 12x memory usage per point
pub struct Windowed2Signed2Data {
    pub windowed2_signed_tables: Vec<Windowed2Signed2Table>,
}

impl Windowed2Signed2Data {
    pub fn new(points: &[G1Projective]) -> Self {
        let windowed2_signed_tables = points
            .par_iter()
            .map(|point| {
                let glv_bases = [*point, glv_endomorphism(point)];
                Windowed2Signed2Table::new(&glv_bases)
            })
            .collect();

        Self {
            windowed2_signed_tables,
        }
    }
}

/// 2D GLV scalar multiplication using precomputed data and decomposed scalar
pub fn glv_two_scalar_mul_decomposed(
    precomputed_data: &PrecomputedShamir2Data,
    scalar_coeffs: &[<Fr as PrimeField>::BigInt; 2],
    scalar_signs: &[bool; 2],
) -> Vec<G1Projective> {
    precomputed_data
        .shamir_tables
        .par_iter()
        .map(|shamir_table| {
            shamir_glv_mul_2d_precomputed(shamir_table, scalar_coeffs, scalar_signs)
        })
        .collect()
}

/// 2D GLV scalar multiplication using precomputed data
pub fn glv_two_scalar_mul(
    precomputed_data: &PrecomputedShamir2Data,
    scalar: Fr,
) -> Vec<G1Projective> {
    let (scalar_coeffs, scalar_signs) = decompose_scalar_2d(scalar);
    glv_two_scalar_mul_decomposed(precomputed_data, &scalar_coeffs, &scalar_signs)
}

/// 2D GLV scalar multiplication online (no precomputation)
pub fn glv_two_scalar_mul_online(scalar: Fr, points: &[G1Projective]) -> Vec<G1Projective> {
    let (scalar_coeffs, scalar_signs) = decompose_scalar_2d(scalar);

    points
        .par_iter()
        .map(|point| {
            let glv_bases = [*point, glv_endomorphism(point)];
            let shamir_table = PrecomputedShamir2Table::new(&glv_bases);
            shamir_glv_mul_2d_precomputed(&shamir_table, &scalar_coeffs, &scalar_signs)
        })
        .collect()
}

/// Decomposed scalar for 2D GLV
#[derive(Clone, Debug)]
pub struct DecomposedScalar2D {
    pub coeffs: [<Fr as PrimeField>::BigInt; 2],
    pub signs: [bool; 2],
}

impl DecomposedScalar2D {
    /// Create from a scalar
    pub fn from_scalar(scalar: Fr) -> Self {
        let (coeffs, signs) = decompose_scalar_2d(scalar);
        Self { coeffs, signs }
    }
}

/// Precomputed data for fixed-base vector MSM in G1
///
/// This structure holds precomputed GLV endomorphism bases and Shamir tables
/// for a fixed base point, allowing efficient multiplication by multiple scalars.
#[derive(Clone, Debug)]
pub struct FixedBasePrecomputedG1 {
    /// The GLV endomorphism bases [P, λ(P)]
    pub glv_bases: [G1Projective; 2],
    /// Precomputed Shamir table for all combinations and signs
    pub shamir_table: PrecomputedShamir2Table,
}

impl FixedBasePrecomputedG1 {
    /// Create precomputed data for a fixed base point
    pub fn new(base: &G1Projective) -> Self {
        let glv_bases = [*base, glv_endomorphism(base)];
        let shamir_table = PrecomputedShamir2Table::new(&glv_bases);

        Self {
            glv_bases,
            shamir_table,
        }
    }

    /// Multiply the fixed base by a single scalar using decomposed form
    pub fn mul_scalar_decomposed(&self, decomposed_scalar: &DecomposedScalar2D) -> G1Projective {
        shamir_glv_mul_2d_precomputed(
            &self.shamir_table,
            &decomposed_scalar.coeffs,
            &decomposed_scalar.signs,
        )
    }

    /// Multiply the fixed base by a single scalar
    pub fn mul_scalar(&self, scalar: Fr) -> G1Projective {
        let decomposed_scalar = DecomposedScalar2D::from_scalar(scalar);
        self.mul_scalar_decomposed(&decomposed_scalar)
    }

    /// Multiply the fixed base by multiple scalars (all decomposed)
    pub fn mul_scalars_decomposed(
        &self,
        decomposed_scalars: &[DecomposedScalar2D],
    ) -> Vec<G1Projective> {
        decomposed_scalars
            .par_iter()
            .map(|decomposed_scalar| self.mul_scalar_decomposed(decomposed_scalar))
            .collect()
    }

    /// Multiply the fixed base by multiple scalars
    pub fn mul_scalars(&self, scalars: &[Fr]) -> Vec<G1Projective> {
        scalars
            .par_iter()
            .map(|scalar| self.mul_scalar(*scalar))
            .collect()
    }
}

/// Fixed-base vector MSM for G1: multiply a single base point by multiple scalars
///
/// This function efficiently computes `base * scalars[i]` for all i using 2D GLV decomposition.
/// It precomputes the GLV endomorphism bases for the fixed base once and reuses them
/// for all scalar multiplications, providing significant speedup compared to naive approaches.
///
/// # Arguments
/// * `base` - The fixed G1 base point to multiply
/// * `scalars` - Vector of scalars to multiply the base by
///
/// # Returns
/// Vector of results where `result[i] = base * scalars[i]`
///
/// # Performance
/// This is optimal when you have a fixed base point and multiple different scalars,
/// as it avoids recomputing GLV endomorphisms for each scalar multiplication.
pub fn fixed_base_vector_msm_g1(base: &G1Projective, scalars: &[Fr]) -> Vec<G1Projective> {
    let precomputed = FixedBasePrecomputedG1::new(base);
    precomputed.mul_scalars(scalars)
}

/// 2-bit signed windowed 2D GLV scalar multiplication using precomputed data
pub fn glv_two_scalar_mul_windowed2_signed(
    windowed2_signed_data: &Windowed2Signed2Data,
    scalar: Fr,
) -> Vec<G1Projective> {
    let (scalar_coeffs, scalar_signs) = decompose_scalar_2d(scalar);

    windowed2_signed_data
        .windowed2_signed_tables
        .par_iter()
        .map(|windowed2_signed_table| {
            shamir_glv_mul_windowed2_signed_2d(
                windowed2_signed_table,
                &scalar_coeffs,
                &scalar_signs,
            )
        })
        .collect()
}

/// Precomputation functions
pub fn glv_two_precompute(points: &[G1Projective]) -> PrecomputedShamir2Data {
    PrecomputedShamir2Data::new(points)
}

pub fn glv_two_precompute_windowed2_signed(points: &[G1Projective]) -> Windowed2Signed2Data {
    Windowed2Signed2Data::new(points)
}

/// Precomputed data for G1 vector scalar multiplication using 2D GLV
pub struct VectorScalarMulG1Data {
    pub precomputed_data: PrecomputedShamir2Data,
    pub scalar_coeffs: [<Fr as PrimeField>::BigInt; 2],
    pub scalar_signs: [bool; 2],
}

impl VectorScalarMulG1Data {
    /// Create precomputed data for vector scalar multiplication with 2D GLV
    pub fn new(generators: &[G1Projective], scalar: Fr) -> Self {
        let precomputed_data = glv_two_precompute(generators);
        let (scalar_coeffs, scalar_signs) = decompose_scalar_2d(scalar);

        Self {
            precomputed_data,
            scalar_coeffs,
            scalar_signs,
        }
    }
}

/// Perform G1 vector scalar multiplication and addition using precomputed data
/// Computes `v[i] = v[i] + scalar * generators[i]` for all i
pub fn vector_scalar_mul_add_g1_precomputed(v: &mut [G1Projective], data: &VectorScalarMulG1Data) {
    use rayon::prelude::*;

    // Perform scalar multiplication and addition in parallel
    v.par_iter_mut().enumerate().for_each(|(i, v_point)| {
        let scalar_mul_result = shamir_glv_mul_2d_precomputed(
            &data.precomputed_data.shamir_tables[i],
            &data.scalar_coeffs,
            &data.scalar_signs,
        );
        *v_point += scalar_mul_result;
    });
}

/// Perform G1 vector scalar multiplication and addition online (no precomputation)
/// Computes `v[i] = v[i] + scalar * generators[i]` for all i
pub fn vector_scalar_mul_add_g1_online(
    v: &mut [G1Projective],
    generators: &[G1Projective],
    scalar: Fr,
) {
    use rayon::prelude::*;

    assert_eq!(
        v.len(),
        generators.len(),
        "v and generators must have same length"
    );

    let (scalar_coeffs, scalar_signs) = decompose_scalar_2d(scalar);

    // Perform scalar multiplication and addition in parallel
    v.par_iter_mut()
        .zip(generators.par_iter())
        .for_each(|(v_point, generator)| {
            let glv_bases = [*generator, glv_endomorphism(generator)];
            let shamir_table = PrecomputedShamir2Table::new(&glv_bases);
            let scalar_mul_result =
                shamir_glv_mul_2d_precomputed(&shamir_table, &scalar_coeffs, &scalar_signs);
            *v_point += scalar_mul_result;
        });
}

/// Convenience function that combines precomputation and execution
/// Computes `v[i] = v[i] + scalar * generators[i]` for all i
pub fn vector_scalar_mul_add_g1(v: &mut [G1Projective], generators: &[G1Projective], scalar: Fr) {
    let data = VectorScalarMulG1Data::new(generators, scalar);
    vector_scalar_mul_add_g1_precomputed(v, &data);
}

/// Precomputed data for efficient vector scalar multiplication where we scale the vector elements
/// and add generators: v[i] = scalar * v[i] + generators[i]
#[derive(Clone, Debug)]
pub struct VectorScalarMulG1VData {
    /// Decomposed scalar coefficients
    pub scalar_coeffs: [<Fr as PrimeField>::BigInt; 2],
    /// Signs for each coefficient
    pub scalar_signs: [bool; 2],
}

impl VectorScalarMulG1VData {
    /// Create precomputed scalar decomposition for vector element scaling
    ///
    /// # Arguments
    /// * `scalar` - Fixed scalar that will be used to scale vector elements
    pub fn new(scalar: Fr) -> Self {
        let (scalar_coeffs, scalar_signs) = decompose_scalar_2d(scalar);

        Self {
            scalar_coeffs,
            scalar_signs,
        }
    }
}

/// Perform vector scalar multiplication with vector scaling using precomputed data
///
/// Computes `v[i] = scalar * v[i] + generators[i]` for all i, where scalar decomposition
/// is precomputed in `data`.
///
/// # Arguments
/// * `v` - Mutable reference to vector to update (will be scaled and then added to)
/// * `generators` - Fixed G1 generators to add to scaled vector elements
/// * `data` - Precomputed data containing decomposed scalar
///
/// # Panics
/// * If `v.len() != generators.len()`
pub fn vector_scalar_mul_v_add_g_g1_precomputed(
    v: &mut [G1Projective],
    generators: &[G1Projective],
    data: &VectorScalarMulG1VData,
) {
    assert_eq!(
        v.len(),
        generators.len(),
        "Vector and generators must have the same length"
    );

    use rayon::prelude::*;

    // Perform scalar multiplication and addition in parallel
    v.par_iter_mut()
        .zip(generators.par_iter())
        .for_each(|(v_point, generator)| {
            // Compute GLV bases for current vector element
            let glv_bases = [*v_point, glv_endomorphism(v_point)];

            // Create temporary Shamir table for v_point
            let shamir_table = PrecomputedShamir2Table::new(&glv_bases);

            // Perform scalar multiplication: scalar * v[i] + generators[i]
            let v_scaled = shamir_glv_mul_2d_precomputed(
                &shamir_table,
                &data.scalar_coeffs,
                &data.scalar_signs,
            );
            *v_point = v_scaled + generator;
        });
}

/// Perform vector scalar multiplication with vector scaling online (without precomputation)
///
/// Computes `v[i] = scalar * v[i] + generators[i]` for all i.
/// This version decomposes the scalar once but doesn't use precomputed tables.
///
/// # Arguments
/// * `v` - Mutable reference to vector to update (will be scaled and then added to)
/// * `generators` - Fixed G1 generators to add to scaled vector elements  
/// * `scalar` - Fixed scalar to multiply with each vector element
///
/// # Panics
/// * If `v.len() != generators.len()`
pub fn vector_scalar_mul_v_add_g_g1_online(
    v: &mut [G1Projective],
    generators: &[G1Projective],
    scalar: Fr,
) {
    assert_eq!(
        v.len(),
        generators.len(),
        "Vector and generators must have the same length"
    );

    let data = VectorScalarMulG1VData::new(scalar);
    vector_scalar_mul_v_add_g_g1_precomputed(v, generators, &data);
}

/// Core Shamir trick implementation for 2D GLV using precomputed table
pub fn shamir_glv_mul_2d_precomputed(
    shamir_table: &PrecomputedShamir2Table,
    scalars: &[<Fr as PrimeField>::BigInt; 2],
    signs: &[bool; 2],
) -> G1Projective {
    // Convert BigInt scalars to unsigned bit arrays (signs handled separately)
    let bit_arrays: Vec<Vec<u8>> = scalars
        .iter()
        .map(|scalar| {
            let mut bits = Vec::new();
            let scalar_ref = scalar.as_ref();

            for limb in scalar_ref.iter() {
                for bit_idx in 0..64 {
                    let bit = ((*limb >> bit_idx) & 1) as u8;
                    bits.push(bit);
                }
            }
            bits
        })
        .collect();

    // Find maximum bit length
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

    let mut result = G1Projective::zero();

    for bit_idx in (0..max_bits).rev() {
        // Double the accumulator
        result = result.double();

        // Extract point mask and sign mask
        let mut point_mask = 0;
        let mut sign_mask = 0;
        for i in 0..2 {
            if bit_idx < bit_arrays[i].len() && bit_arrays[i][bit_idx] == 1 {
                point_mask |= 1 << i;
                // Apply decomposition sign: if signs[i] is false, negate this component
                if !signs[i] {
                    sign_mask |= 1 << i;
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

/// 2-bit signed windowed Shamir trick for 2D GLV using Windowed2Signed2Table
pub fn shamir_glv_mul_windowed2_signed_2d(
    windowed2_signed_table: &Windowed2Signed2Table,
    scalars: &[<Fr as PrimeField>::BigInt; 2],
    signs: &[bool; 2],
) -> G1Projective {
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
    let mut result = G1Projective::zero();

    for coeff_idx in (0..max_coeffs).rev() {
        // Quadruple (process 2 bits): result = 4 * result
        result = result.double().double();

        // Extract coefficients for this window and use signed table lookup
        let mut window_coeffs = [0i8; 2];
        for i in 0..2 {
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
