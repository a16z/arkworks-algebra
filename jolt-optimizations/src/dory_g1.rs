//! Dory protocol vector operations for G1
//! Implements v[i] = v[i] + scalar * g[i] ; v[i] = scalar * v[i] + gamma[i]
//! Using GLV + strauss-shamir's trick extended to 4 scalars
//! https://crypto.stackexchange.com/questions/99975/strauss-shamir-trick-on-ec-multiplication-by-scalar

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::CurveGroup;
use rayon::prelude::*;

use crate::decomp_2d::{decompose_scalar_2d, glv_endomorphism_affine};
use crate::glv_two::{shamir_glv_mul_2d_affine, shamir_glv_mul_2d_precomputed};
use crate::{
    glv_two_precompute, glv_two_precompute_windowed2_signed, glv_two_scalar_mul_online,
    glv_two_scalar_mul_windowed2_signed, PrecomputedShamir2Data, PrecomputedShamir2Table,
    Windowed2Signed2Data,
};

/// Minimum vector length to justify rayon par_iter overhead
const MIN_PAR_SIZE: usize = 64;


/// Online version — batch-normalizes generators to avoid per-element inversions
pub fn vector_add_scalar_mul_g1_online(
    v: &mut [G1Projective],
    generators: &[G1Projective],
    scalar: Fr,
) {
    assert_eq!(v.len(), generators.len());
    let (coeffs, signs) = decompose_scalar_2d(scalar);

    let affine_gens = G1Projective::normalize_batch(generators);

    let body = |(vi, affine_gen): (&mut G1Projective, &G1Affine)| {
        let glv_affine = glv_endomorphism_affine(affine_gen);
        *vi += shamir_glv_mul_2d_affine(&[*affine_gen, glv_affine], &coeffs, &signs);
    };

    if v.len() >= MIN_PAR_SIZE {
        v.par_iter_mut()
            .zip(affine_gens.par_iter())
            .for_each(body);
    } else {
        v.iter_mut().zip(affine_gens.iter()).for_each(body);
    }
}

/// Precomputed full
pub fn vector_add_scalar_mul_g1_precomputed(
    v: &mut [G1Projective],
    scalar: Fr,
    precomputed_tables: &[PrecomputedShamir2Table],
) {
    assert_eq!(v.len(), precomputed_tables.len());
    let (coeffs, signs) = decompose_scalar_2d(scalar);

    let body = |(vi, table): (&mut G1Projective, &PrecomputedShamir2Table)| {
        *vi += shamir_glv_mul_2d_precomputed(table, &coeffs, &signs);
    };

    if v.len() >= MIN_PAR_SIZE {
        v.par_iter_mut()
            .zip(precomputed_tables.par_iter())
            .for_each(body);
    } else {
        v.iter_mut().zip(precomputed_tables.iter()).for_each(body);
    }
}

/// 2-bit signed precomputed
pub fn vector_add_scalar_mul_g1_windowed2_signed(
    v: &mut [G1Projective],
    scalar: Fr,
    precomputed_generators: &Windowed2Signed2Data,
) {
    assert_eq!(v.len(), precomputed_generators.windowed2_tables.len());

    let products = glv_two_scalar_mul_windowed2_signed(precomputed_generators, scalar);

    let body = |(vi, &prod): (&mut G1Projective, &G1Projective)| {
        *vi += prod;
    };

    if v.len() >= MIN_PAR_SIZE {
        v.par_iter_mut().zip(products.par_iter()).for_each(body);
    } else {
        v.iter_mut().zip(products.iter()).for_each(body);
    }
}


/// Online — batch-normalizes all v[i] in one shot to avoid per-element inversions
pub fn vector_scalar_mul_add_gamma_g1_online(
    v: &mut [G1Projective],
    scalar: Fr,
    gamma: &[G1Projective],
) {
    assert_eq!(v.len(), gamma.len());
    let (coeffs, signs) = decompose_scalar_2d(scalar);

    let affine_v = G1Projective::normalize_batch(v);

    let body = |((vi, affine_vi), &gamma_i): ((&mut G1Projective, &G1Affine), &G1Projective)| {
        let glv_affine = glv_endomorphism_affine(affine_vi);
        *vi = shamir_glv_mul_2d_affine(&[*affine_vi, glv_affine], &coeffs, &signs) + gamma_i;
    };

    if v.len() >= MIN_PAR_SIZE {
        v.par_iter_mut()
            .zip(affine_v.par_iter())
            .zip(gamma.par_iter())
            .for_each(body);
    } else {
        v.iter_mut()
            .zip(affine_v.iter())
            .zip(gamma.iter())
            .for_each(body);
    }
}

pub fn vector_scalar_mul_add_gamma_g1_precomputed(
    v: &mut [G1Projective],
    scalar: Fr,
    gamma: &[G1Projective],
) {
    vector_scalar_mul_add_gamma_g1_online(v, scalar, gamma);
}

/// 2-bit signed precomputed
/// Note: We can't precompute on v since it changes, so this uses online scalar mul
pub fn vector_scalar_mul_add_gamma_g1_windowed2_signed(
    v: &mut [G1Projective],
    scalar: Fr,
    gamma: &[G1Projective],
) {
    assert_eq!(v.len(), gamma.len());

    let products = glv_two_scalar_mul_online(scalar, v);

    let body = |((vi, &prod), &gamma_i): ((&mut G1Projective, &G1Projective), &G1Projective)| {
        *vi = prod + gamma_i;
    };

    if v.len() >= MIN_PAR_SIZE {
        v.par_iter_mut()
            .zip(products.par_iter())
            .zip(gamma.par_iter())
            .for_each(body);
    } else {
        v.iter_mut()
            .zip(products.iter())
            .zip(gamma.iter())
            .for_each(body);
    }
}


/// Precompute Shamir tables for a set of G1 generators
pub fn precompute_g1_generators(generators: &[G1Projective]) -> PrecomputedShamir2Data {
    glv_two_precompute(generators)
}

/// Precompute 2-bit signed tables for a set of G1 generators
pub fn precompute_g1_generators_windowed2_signed(
    generators: &[G1Projective],
) -> Windowed2Signed2Data {
    glv_two_precompute_windowed2_signed(generators)
}
