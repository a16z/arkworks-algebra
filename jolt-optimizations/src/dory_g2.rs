//! Dory protocol vector operations for G2
//! Implements v[i] = v[i] + scalar * g[i] and v[i] = scalar * v[i] + gamma[i]

use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::CurveGroup;
use rayon::prelude::*;

use crate::decomp_4d::decompose_scalar_4d;
use crate::frobenius::frobenius_psi_power_affine;
use crate::glv_four::{shamir_glv_mul_4d_affine, shamir_glv_mul_4d_precomputed};
use crate::{
    glv_four_precompute, glv_four_precompute_windowed2_signed, glv_four_scalar_mul_online,
    glv_four_scalar_mul_windowed2_signed, PrecomputedShamir4Data, PrecomputedShamir4Table,
    Windowed2Signed4Data,
};

/// Minimum vector length to justify rayon par_iter overhead
const MIN_PAR_SIZE: usize = 64;


/// Online version — batch-normalizes generators, uses affine Frobenius + Shamir
pub fn vector_add_scalar_mul_g2_online(
    v: &mut [G2Projective],
    generators: &[G2Projective],
    scalar: Fr,
) {
    assert_eq!(v.len(), generators.len());
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    let affine_gens = G2Projective::normalize_batch(generators);

    let body = |(vi, affine_gen): (&mut G2Projective, &G2Affine)| {
        let bases = [
            *affine_gen,
            frobenius_psi_power_affine(affine_gen, 1),
            frobenius_psi_power_affine(affine_gen, 2),
            frobenius_psi_power_affine(affine_gen, 3),
        ];
        *vi += shamir_glv_mul_4d_affine(&bases, &coeffs, &signs);
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
pub fn vector_add_scalar_mul_g2_precomputed(
    v: &mut [G2Projective],
    scalar: Fr,
    precomputed_tables: &[PrecomputedShamir4Table],
) {
    assert_eq!(v.len(), precomputed_tables.len());
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    let body = |(vi, table): (&mut G2Projective, &PrecomputedShamir4Table)| {
        *vi += shamir_glv_mul_4d_precomputed(table, &coeffs, &signs);
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
pub fn vector_add_scalar_mul_g2_windowed2_signed(
    v: &mut [G2Projective],
    scalar: Fr,
    precomputed_generators: &Windowed2Signed4Data,
) {
    assert_eq!(v.len(), precomputed_generators.windowed2_tables.len());

    let products = glv_four_scalar_mul_windowed2_signed(precomputed_generators, scalar);

    let body = |(vi, &prod): (&mut G2Projective, &G2Projective)| {
        *vi += prod;
    };

    if v.len() >= MIN_PAR_SIZE {
        v.par_iter_mut().zip(products.par_iter()).for_each(body);
    } else {
        v.iter_mut().zip(products.iter()).for_each(body);
    }
}


/// Online — batch-normalizes v, uses affine Frobenius + Shamir
pub fn vector_scalar_mul_add_gamma_g2_online(
    v: &mut [G2Projective],
    scalar: Fr,
    gamma: &[G2Projective],
) {
    assert_eq!(v.len(), gamma.len());
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    let affine_v = G2Projective::normalize_batch(v);

    let body = |((vi, affine_vi), &gamma_i): ((&mut G2Projective, &G2Affine), &G2Projective)| {
        let bases = [
            *affine_vi,
            frobenius_psi_power_affine(affine_vi, 1),
            frobenius_psi_power_affine(affine_vi, 2),
            frobenius_psi_power_affine(affine_vi, 3),
        ];
        *vi = shamir_glv_mul_4d_affine(&bases, &coeffs, &signs) + gamma_i;
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

/// Precomputed full
pub fn vector_scalar_mul_add_gamma_g2_precomputed(
    v: &mut [G2Projective],
    scalar: Fr,
    gamma: &[G2Projective],
) {
    vector_scalar_mul_add_gamma_g2_online(v, scalar, gamma);
}

pub fn vector_scalar_mul_add_gamma_g2_windowed2_signed(
    v: &mut [G2Projective],
    scalar: Fr,
    gamma: &[G2Projective],
) {
    assert_eq!(v.len(), gamma.len());

    let products = glv_four_scalar_mul_online(scalar, v);

    let body = |((vi, &prod), &gamma_i): ((&mut G2Projective, &G2Projective), &G2Projective)| {
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


/// Precompute Shamir tables for a set of G2 generators
pub fn precompute_g2_generators(generators: &[G2Projective]) -> PrecomputedShamir4Data {
    glv_four_precompute(generators)
}

/// Precompute 2-bit signed tables for a set of G2 generators
pub fn precompute_g2_generators_windowed2_signed(
    generators: &[G2Projective],
) -> Windowed2Signed4Data {
    glv_four_precompute_windowed2_signed(generators)
}
