//! 4-dimensional scalar decomposition for BN254 G2
//!
//! Decomposes a scalar k into k = k0 + k1*λ + k2*λ² + k3*λ³
//! where λ is the Frobenius endomorphism eigenvalue.
//! Uses precomputed table of power-of-2 decompositions for zero-allocation operation.

use crate::constants::POWER_OF_2_DECOMPOSITIONS;
use ark_bn254::Fr;
use ark_ff::PrimeField;

/// Decompose scalar for 4D GLV multiplication.
/// Zero-allocation: operates on raw u64 limbs via trailing_zeros bit scanning.
pub fn decompose_scalar_4d(scalar: Fr) -> ([<Fr as PrimeField>::BigInt; 4], [bool; 4]) {
    let limbs = scalar.into_bigint().0;

    let mut k = [0i128; 4];

    for limb_idx in 0..4 {
        let limb = limbs[limb_idx];
        if limb == 0 {
            continue;
        }
        let base_bit = limb_idx * 64;
        let mut remaining = limb;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            let bit_pos = base_bit + bit;

            let (d0, d1, d2, d3, n0, n1, n2, n3) =
                POWER_OF_2_DECOMPOSITIONS[bit_pos];

            k[0] += if n0 { -(d0 as i128) } else { d0 as i128 };
            k[1] += if n1 { -(d1 as i128) } else { d1 as i128 };
            k[2] += if n2 { -(d2 as i128) } else { d2 as i128 };
            k[3] += if n3 { -(d3 as i128) } else { d3 as i128 };

            remaining &= remaining - 1;
        }
    }

    let mut coeffs = [<Fr as PrimeField>::BigInt::default(); 4];
    let mut signs = [false; 4];

    for i in 0..4 {
        if k[i] < 0 {
            signs[i] = true;
            coeffs[i].0[0] = (-k[i]) as u64;
            coeffs[i].0[1] = ((-k[i]) >> 64) as u64;
        } else {
            coeffs[i].0[0] = k[i] as u64;
            coeffs[i].0[1] = (k[i] >> 64) as u64;
        }
    }

    (coeffs, signs)
}
