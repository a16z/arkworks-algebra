//! 2D GLV scalar multiplication implementations for BN254 G1
//!
//! This module provides optimized scalar multiplication algorithms for BN254 G1 using
//! 2-dimensional GLV decomposition with the Shamir trick and precomputed lookup tables.

use ark_bn254::{Fq, Fr, G1Projective};
use ark_ec::{AdditiveGroup, PrimeGroup};
use ark_ff::{BigInteger, MontFp, PrimeField};
use ark_std::ops::{AddAssign, Neg};
use ark_std::Zero;
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed};
use rayon::prelude::*;

/// GLV lambda for BN254 G1 (from arkworks v0.5)
const LAMBDA: Fr =
    MontFp!("21888242871839275217838484774961031246154997185409878258781734729429964517155");

/// GLV endomorphism coefficient for BN254 G1 (from arkworks v0.5)
const ENDO_COEFF: Fq =
    MontFp!("21888242871839275220042445260109153167277707414472061641714758635765020556616");

/// GLV scalar decomposition coefficients for BN254 G1 (from arkworks v0.5)
const SCALAR_DECOMP_COEFFS: [(bool, <Fr as PrimeField>::BigInt); 4] = [
    (
        false,
        ark_ff::BigInt!("147946756881789319000765030803803410728"),
    ),
    (true, ark_ff::BigInt!("9931322734385697763")),
    (false, ark_ff::BigInt!("9931322734385697763")),
    (
        false,
        ark_ff::BigInt!("147946756881789319010696353538189108491"),
    ),
];

/// Helper function to decompose a scalar using 2D GLV (arkworks v0.5 implementation)
fn decompose_scalar_2d(scalar: Fr) -> ([<Fr as PrimeField>::BigInt; 2], [bool; 2]) {
    // Convert to num_bigint::BigInt for arithmetic
    let scalar_bytes = scalar.into_bigint().to_bytes_be();
    let scalar_bigint = BigInt::from_bytes_be(Sign::Plus, &scalar_bytes);

    let coeff_bigints: [BigInt; 4] = SCALAR_DECOMP_COEFFS.map(|x| {
        let bytes = x.1.to_bytes_be();
        BigInt::from_bytes_be(x.0.then_some(Sign::Plus).unwrap_or(Sign::Minus), &bytes)
    });

    let [n11, n12, n21, n22] = coeff_bigints;

    let r_bytes = Fr::MODULUS.to_bytes_be();
    let r = BigInt::from_bytes_be(Sign::Plus, &r_bytes);

    // beta = vector([k,0]) * self.curve.N_inv
    // The inverse of N is 1/r * Matrix([[n22, -n12], [-n21, n11]]).
    // so β = (k*n22, -k*n12)/r

    let beta_1 = {
        let (mut div, rem) = (&scalar_bigint * &n22).div_rem(&r);
        if (&rem + &rem) > r {
            div.add_assign(BigInt::one());
        }
        div
    };
    let beta_2 = {
        let (mut div, rem) = (&scalar_bigint * &n12.clone().neg()).div_rem(&r);
        if (&rem + &rem) > r {
            div.add_assign(BigInt::one());
        }
        div
    };

    // b = vector([int(beta[0]), int(beta[1])]) * self.curve.N
    // b = (β1N11 + β2N21, β1N12 + β2N22) with the signs!
    //   = (b11   + b12  , b21   + b22)   with the signs!

    // b1
    let b11 = &beta_1 * &n11;
    let b12 = &beta_2 * &n21;
    let b1 = b11 + b12;

    // b2
    let b21 = &beta_1 * &n12;
    let b22 = &beta_2 * &n22;
    let b2 = b21 + b22;

    let k1 = &scalar_bigint - b1;
    let k1_abs = BigUint::try_from(k1.abs()).unwrap();

    // k2
    let k2 = -b2;
    let k2_abs = BigUint::try_from(k2.abs()).unwrap();

    // Convert back to arkworks BigInt
    let k1_fr = Fr::from(k1_abs);
    let k2_fr = Fr::from(k2_abs);

    let k_bigint = [k1_fr.into_bigint(), k2_fr.into_bigint()];

    let signs = [k1.sign() == Sign::Plus, k2.sign() == Sign::Plus];

    (k_bigint, signs)
}

/// Apply GLV endomorphism to G1 point (arkworks v0.5 implementation)
fn glv_endomorphism(point: &G1Projective) -> G1Projective {
    let mut res = *point;
    res.x *= ENDO_COEFF;
    res
}

/// Precomputed Shamir lookup table for 2-point scalar multiplication with signed combinations
/// Contains all 16 combinations: 4 point combinations × 4 sign patterns
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::G1Affine;
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_simple_glv_manual() {
        let mut rng = test_rng();
        let point = G1Affine::rand(&mut rng).into_group();
        let scalar = Fr::rand(&mut rng);

        // Get decomposition
        let (coeffs, signs) = decompose_scalar_2d(scalar);

        // Manual GLV multiplication: scalar * P = k1 * P + k2 * λ(P)
        let k1 = Fr::from_bigint(coeffs[0]).unwrap();
        let k2 = Fr::from_bigint(coeffs[1]).unwrap();

        let k1_signed = if signs[0] { k1 } else { -k1 };
        let k2_signed = if signs[1] { k2 } else { -k2 };

        let p1 = point.mul_bigint(k1_signed.into_bigint());
        let p2 = glv_endomorphism(&point).mul_bigint(k2_signed.into_bigint());
        let glv_result = p1 + p2;

        let expected = point.mul_bigint(scalar.into_bigint());

        assert_eq!(
            glv_result.into_affine(),
            expected.into_affine(),
            "Manual GLV failed: scalar={:?}, k1={:?} (sign={}), k2={:?} (sign={})",
            scalar,
            k1,
            signs[0],
            k2,
            signs[1]
        );
    }

    #[test]
    fn test_glv_decomposition_algebraic_correctness() {
        let mut rng = test_rng();

        // Test multiple random scalars
        for _ in 0..10 {
            let scalar = Fr::rand(&mut rng);
            let (coeffs, signs) = decompose_scalar_2d(scalar);

            // Reconstruct the scalar from decomposition: k = k1 + λ*k2
            let k1 = Fr::from_bigint(coeffs[0]).unwrap();
            let k2 = Fr::from_bigint(coeffs[1]).unwrap();

            // Apply signs
            let k1_signed = if signs[0] { k1 } else { -k1 };
            let k2_signed = if signs[1] { k2 } else { -k2 };

            // Verify: scalar = k1 + λ*k2
            let reconstructed = k1_signed + LAMBDA * k2_signed;

            assert_eq!(
                scalar, reconstructed,
                "GLV decomposition failed: scalar={:?}, k1={:?}, k2={:?}, signs={:?}",
                scalar, k1, k2, signs
            );
        }
    }

    #[test]
    fn test_glv_two_consistency() {
        let mut rng = test_rng();

        // Generate test data
        let num_points = 5;
        let points: Vec<G1Projective> = (0..num_points)
            .map(|_| G1Affine::rand(&mut rng).into_group())
            .collect();
        let scalar = Fr::rand(&mut rng);

        // Test online version
        let result_online = glv_two_scalar_mul_online(scalar, &points);

        // Test precomputed version
        let precomputed_data = glv_two_precompute(&points);
        let result_precomputed = glv_two_scalar_mul(&precomputed_data, scalar);

        // Test windowed2 signed version
        let windowed2_signed_data = glv_two_precompute_windowed2_signed(&points);
        let result_windowed2_signed =
            glv_two_scalar_mul_windowed2_signed(&windowed2_signed_data, scalar);

        // Compare with naive scalar multiplication
        for i in 0..num_points {
            let expected = points[i].mul_bigint(scalar.into_bigint());

            // Convert to affine for comparison
            let expected_affine = expected.into_affine();
            let online_affine = result_online[i].into_affine();
            // let precomputed_affine = result_precomputed[i].into_affine();
            // let windowed2_signed_affine = result_windowed2_signed[i].into_affine();

            assert_eq!(
                expected_affine, online_affine,
                "Online version mismatch at index {}",
                i
            );
            // assert_eq!(expected_affine, precomputed_affine, "Precomputed version mismatch at index {}", i);
            // assert_eq!(expected_affine, windowed2_signed_affine, "Windowed2 signed version mismatch at index {}", i);
        }
    }

    // #[test]
    // fn test_edge_cases_2d() {
    //     let mut rng = test_rng();

    //     // Test with single point
    //     let points = vec![G1Affine::rand(&mut rng).into_group()];
    //     let scalar = Fr::rand(&mut rng);

    //     let result_online = glv_two_scalar_mul_online(scalar, &points);

    //     let expected = points[0].mul_bigint(scalar.into_bigint());
    //     assert_eq!(result_online[0].into_affine(), expected.into_affine());

    //     // Test with zero scalar
    //     let scalar_zero = Fr::from(0u64);
    //     let result_zero = glv_two_scalar_mul_online(scalar_zero, &points);
    //     assert_eq!(result_zero[0], G1Projective::zero());

    //     // Test with identity point
    //     let identity_points = vec![G1Projective::zero()];
    //     let result_identity = glv_two_scalar_mul_online(scalar, &identity_points);
    //     assert_eq!(result_identity[0], G1Projective::zero());
    // }

    #[test]
    fn test_g1_vector_scalar_mul_add() {
        let mut rng = test_rng();

        // Generate test data
        let num_points = 10;
        let generators: Vec<G1Projective> = (0..num_points)
            .map(|_| G1Affine::rand(&mut rng).into_group())
            .collect();
        let scalar = Fr::rand(&mut rng);

        // Initialize v with random points
        let mut v_online: Vec<G1Projective> = (0..num_points)
            .map(|_| G1Affine::rand(&mut rng).into_group())
            .collect();
        let mut v_precomputed = v_online.clone();
        let mut v_convenience = v_online.clone();
        let v_original = v_online.clone();

        // Test online version
        vector_scalar_mul_add_g1_online(&mut v_online, &generators, scalar);

        // Test precomputed version
        let data = VectorScalarMulG1Data::new(&generators, scalar);
        vector_scalar_mul_add_g1_precomputed(&mut v_precomputed, &data);

        // Test convenience function
        vector_scalar_mul_add_g1(&mut v_convenience, &generators, scalar);

        // Compare with naive computation
        for i in 0..num_points {
            let expected = v_original[i] + generators[i].mul_bigint(scalar.into_bigint());

            assert_eq!(
                v_online[i].into_affine(),
                expected.into_affine(),
                "Online version mismatch at index {}",
                i
            );
            assert_eq!(
                v_precomputed[i].into_affine(),
                expected.into_affine(),
                "Precomputed version mismatch at index {}",
                i
            );
            assert_eq!(
                v_convenience[i].into_affine(),
                expected.into_affine(),
                "Convenience version mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_g1_vector_edge_cases() {
        let mut rng = test_rng();

        // Test with single point
        let generators = vec![G1Affine::rand(&mut rng).into_group()];
        let mut v = vec![G1Affine::rand(&mut rng).into_group()];
        let v_original = v[0];
        let scalar = Fr::rand(&mut rng);

        vector_scalar_mul_add_g1_online(&mut v, &generators, scalar);
        let expected = v_original + generators[0].mul_bigint(scalar.into_bigint());
        assert_eq!(v[0].into_affine(), expected.into_affine());

        // Test with zero scalar
        let mut v_zero = vec![v_original];
        let scalar_zero = Fr::from(0u64);
        vector_scalar_mul_add_g1_online(&mut v_zero, &generators, scalar_zero);
        assert_eq!(v_zero[0], v_original);

        // Test with identity generator
        let identity_generators = vec![G1Projective::zero()];
        let mut v_identity = vec![v_original];
        vector_scalar_mul_add_g1_online(&mut v_identity, &identity_generators, scalar);
        assert_eq!(v_identity[0], v_original);
    }
}
