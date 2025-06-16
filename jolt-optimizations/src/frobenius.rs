//! Frobenius endomorphism operations for BN254 G2
//!
//! This module provides functionality for computing Frobenius endomorphisms
//! ψ^k on BN254 G2 curve points.

use crate::constants::get_frobenius_coefficients;
use ark_bn254::{G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::Field;
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

    // Apply the appropriate coefficients based on power
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
pub fn frobenius_psi_power_affine(p: &G2Affine, k: usize) -> G2Affine {
    let projective_result = frobenius_psi_power_projective(&p.into_group(), k);
    projective_result.into_affine()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::AffineRepr;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_frobenius_identity() {
        let mut rng = test_rng();
        let p = G2Affine::rand(&mut rng).into_group();

        // ψ^4 should be the identity
        let psi4_p = frobenius_psi_power_projective(&p, 4);
        assert_eq!(p, psi4_p);

        // ψ^0 should also be the identity
        let psi0_p = frobenius_psi_power_projective(&p, 0);
        assert_eq!(p, psi0_p);
    }

    #[test]
    fn test_frobenius_composition() {
        let mut rng = test_rng();
        let p = G2Affine::rand(&mut rng).into_group();

        // ψ^2 = ψ ∘ ψ
        let psi1_p = frobenius_psi_power_projective(&p, 1);
        let psi1_psi1_p = frobenius_psi_power_projective(&psi1_p, 1);
        let psi2_p = frobenius_psi_power_projective(&p, 2);

        assert_eq!(psi1_psi1_p, psi2_p);
    }

    #[test]
    fn test_frobenius_zero_point() {
        let zero = G2Projective::zero();

        for k in 0..8 {
            let result = frobenius_psi_power_projective(&zero, k);
            assert_eq!(result, zero);
        }
    }
}
