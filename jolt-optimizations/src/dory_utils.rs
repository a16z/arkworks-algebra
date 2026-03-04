//! Dory-specific optimized utilities for vector scalar multiplication

use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::decomp_4d::decompose_scalar_4d;
use crate::frobenius::frobenius_psi_power_affine;
use crate::glv_four::{
    shamir_glv_mul_4d_affine, shamir_glv_mul_4d_precomputed, PrecomputedShamir4Data,
};

/// Minimum vector length to justify rayon par_iter overhead
const MIN_PAR_SIZE: usize = 64;

/// Precomputed data for efficient vector scalar multiplication with a fixed scalar
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct VectorScalarMulData {
    pub scalar_coeffs: [<Fr as PrimeField>::BigInt; 4],
    pub scalar_signs: [bool; 4],
    pub precomputed_data: PrecomputedShamir4Data,
}

impl VectorScalarMulData {
    pub fn new(generators: &[G2Projective], scalar: Fr) -> Self {
        let (scalar_coeffs, scalar_signs) = decompose_scalar_4d(scalar);
        let precomputed_data = PrecomputedShamir4Data::new(generators);

        Self {
            scalar_coeffs,
            scalar_signs,
            precomputed_data,
        }
    }

    pub fn num_generators(&self) -> usize {
        self.precomputed_data.shamir_tables.len()
    }
}

/// v[i] += scalar * generators[i] using precomputed 256-entry Shamir tables
pub fn vector_scalar_mul_add_precomputed(v: &mut [G2Projective], data: &VectorScalarMulData) {
    assert_eq!(v.len(), data.num_generators());

    use rayon::prelude::*;

    let body = |(i, v_point): (usize, &mut G2Projective)| {
        let scalar_mul_result = shamir_glv_mul_4d_precomputed(
            &data.precomputed_data.shamir_tables[i],
            &data.scalar_coeffs,
            &data.scalar_signs,
        );
        *v_point += scalar_mul_result;
    };

    if v.len() >= MIN_PAR_SIZE {
        v.par_iter_mut().enumerate().for_each(body);
    } else {
        v.iter_mut().enumerate().for_each(body);
    }
}

/// v[i] += scalar * generators[i] — batch-normalizes generators, uses affine Shamir
pub fn vector_scalar_mul_add_online(
    v: &mut [G2Projective],
    generators: &[G2Projective],
    scalar: Fr,
) {
    assert_eq!(v.len(), generators.len());

    use rayon::prelude::*;

    let (scalar_coeffs, scalar_signs) = decompose_scalar_4d(scalar);
    let affine_gens = G2Projective::normalize_batch(generators);

    let body = |(v_point, affine_gen): (&mut G2Projective, &G2Affine)| {
        let frobenius_bases = [
            *affine_gen,
            frobenius_psi_power_affine(affine_gen, 1),
            frobenius_psi_power_affine(affine_gen, 2),
            frobenius_psi_power_affine(affine_gen, 3),
        ];
        *v_point += shamir_glv_mul_4d_affine(&frobenius_bases, &scalar_coeffs, &scalar_signs);
    };

    if v.len() >= MIN_PAR_SIZE {
        v.par_iter_mut()
            .zip(affine_gens.par_iter())
            .for_each(body);
    } else {
        v.iter_mut().zip(affine_gens.iter()).for_each(body);
    }
}

/// Convenience: v[i] += scalar * generators[i]
pub fn vector_scalar_mul_add(v: &mut [G2Projective], generators: &[G2Projective], scalar: Fr) {
    let data = VectorScalarMulData::new(generators, scalar);
    vector_scalar_mul_add_precomputed(v, &data);
}

/// Precomputed scalar decomposition for v[i] = scalar * v[i] + generators[i]
#[derive(Clone, Debug)]
pub struct VectorScalarMulVData {
    pub scalar_coeffs: [<Fr as PrimeField>::BigInt; 4],
    pub scalar_signs: [bool; 4],
}

impl VectorScalarMulVData {
    pub fn new(scalar: Fr) -> Self {
        let (scalar_coeffs, scalar_signs) = decompose_scalar_4d(scalar);
        Self {
            scalar_coeffs,
            scalar_signs,
        }
    }
}

/// v[i] = scalar * v[i] + generators[i] — batch-normalizes v, uses affine Frobenius + Shamir
pub fn vector_scalar_mul_v_add_g_precomputed(
    v: &mut [G2Projective],
    generators: &[G2Projective],
    data: &VectorScalarMulVData,
) {
    assert_eq!(v.len(), generators.len());

    use rayon::prelude::*;

    let affine_v = G2Projective::normalize_batch(v);

    let body = |((v_point, affine_vi), generator): (
        (&mut G2Projective, &G2Affine),
        &G2Projective,
    )| {
        let frobenius_bases = [
            *affine_vi,
            frobenius_psi_power_affine(affine_vi, 1),
            frobenius_psi_power_affine(affine_vi, 2),
            frobenius_psi_power_affine(affine_vi, 3),
        ];
        let v_scaled =
            shamir_glv_mul_4d_affine(&frobenius_bases, &data.scalar_coeffs, &data.scalar_signs);
        *v_point = v_scaled + generator;
    };

    if v.len() >= MIN_PAR_SIZE {
        v.par_iter_mut()
            .zip(affine_v.par_iter())
            .zip(generators.par_iter())
            .for_each(body);
    } else {
        v.iter_mut()
            .zip(affine_v.iter())
            .zip(generators.iter())
            .for_each(body);
    }
}

/// v[i] = scalar * v[i] + generators[i] (online)
pub fn vector_scalar_mul_v_add_g_online(
    v: &mut [G2Projective],
    generators: &[G2Projective],
    scalar: Fr,
) {
    assert_eq!(v.len(), generators.len());
    let data = VectorScalarMulVData::new(scalar);
    vector_scalar_mul_v_add_g_precomputed(v, generators, &data);
}
