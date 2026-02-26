//! Frobenius endomorphism operations for BN254 G2
//!
//! This module provides functionality for computing Frobenius endomorphisms
//! ψ^k on BN254 G2 curve points.

use crate::constants::get_frobenius_coefficients;
use ark_bn254::{G2Affine, G2Projective};
use ark_ec::AffineRepr;
use ark_std::Zero;

/// Compute the Frobenius endomorphism ψ^k for BN254 G2 (projective version)
///
/// Implements the Frobenius endomorphism manually as conjugation in Fq2
/// followed by multiplication by precomputed coefficients.
pub fn frobenius_psi_power_projective(p: &G2Projective, k: usize) -> G2Projective {
    if p.is_zero() {
        return *p;
    }

    let mut res = *p;
    let coeffs = get_frobenius_coefficients();

    // Apply Frobenius map to coordinates (conjugation in Fq2)
    if (k & 1) == 1 {
        // Odd power - apply conjugation
        res.x.conjugate_in_place();
        res.y.conjugate_in_place();
        res.z.conjugate_in_place();
    }
    // Even power - identity (no conjugation)

    // Apply the coefficients based on power
    match k % 4 {
        0 => res,
        1 => {
            res.x *= coeffs.psi1_coef2;
            res.y *= coeffs.psi1_coef3;
            res
        },
        2 => {
            res.x *= coeffs.psi2_coef2;
            res.y *= coeffs.psi2_coef3;
            res
        },
        3 => {
            res.x *= coeffs.psi3_coef2;
            res.y *= coeffs.psi3_coef3;
            res
        },
        _ => unreachable!(),
    }
}

/// Compute the Frobenius endomorphism ψ^k for BN254 G2 (affine version)
/// Operates directly on affine coordinates — avoids unnecessary projective conversion.
pub fn frobenius_psi_power_affine(p: &G2Affine, k: usize) -> G2Affine {
    if p.is_zero() {
        return *p;
    }

    let mut x = p.x;
    let mut y = p.y;
    let coeffs = get_frobenius_coefficients();

    if (k & 1) == 1 {
        x.conjugate_in_place();
        y.conjugate_in_place();
    }

    match k % 4 {
        0 => *p,
        1 => G2Affine::new_unchecked(x * coeffs.psi1_coef2, y * coeffs.psi1_coef3),
        2 => G2Affine::new_unchecked(x * coeffs.psi2_coef2, y * coeffs.psi2_coef3),
        3 => G2Affine::new_unchecked(x * coeffs.psi3_coef2, y * coeffs.psi3_coef3),
        _ => unreachable!(),
    }
}
