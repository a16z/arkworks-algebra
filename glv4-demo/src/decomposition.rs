//! 4-dimensional scalar decomposition for BN254 G2
//!
//! This module implements the table-based scalar decomposition algorithm
//! that decomposes a scalar k into k = k0 + k1*λ + k2*λ² + k3*λ³
//! where λ is the Frobenius eigenvalue.

use crate::constants::{get_bn254_frobenius_eigenvalue, POWER_OF_2_DECOMPOSITIONS};
use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use ark_std::Zero;
use num_bigint::{BigInt, Sign};

/// Convert u128 to Fr field element
pub fn u128_to_fr(val: u128) -> Fr {
    let bytes = val.to_be_bytes();
    Fr::from_be_bytes_mod_order(&bytes)
}

/// Convert Fr field element to BigInt
pub fn fr_to_bigint(fr: Fr) -> BigInt {
    let bytes = fr.into_bigint().to_bytes_be();
    BigInt::from_bytes_be(Sign::Plus, &bytes)
}

/// Convert BigInt to Fr field element
pub fn bigint_to_fr(big: &BigInt) -> Fr {
    let (sign, bytes) = big.to_bytes_be();

    if sign == Sign::Minus {
        let r_bytes = Fr::MODULUS.to_bytes_be();
        let r_big = BigInt::from_bytes_be(Sign::Plus, &r_bytes);
        let positive_big = r_big - BigInt::from_bytes_be(Sign::Plus, &bytes);
        let (_sign, positive_bytes) = positive_big.to_bytes_be();
        Fr::from_be_bytes_mod_order(&positive_bytes)
    } else {
        Fr::from_be_bytes_mod_order(&bytes)
    }
}

/// Table-based 4-dimensional scalar decomposition
///
/// Decomposes a scalar k into (k0, k1, k2, k3) such that:
/// k ≡ k0 + k1*λ + k2*λ² + k3*λ³ (mod r)
///
/// Returns the coefficients and their sign flags.
/// Each coefficient is guaranteed to be at most ~66 bits.
pub fn decompose_scalar_table_based(scalar: &BigInt) -> ([u128; 4], [bool; 4]) {
    let mut k0 = 0u128;
    let mut k1 = 0u128;
    let mut k2 = 0u128;
    let mut k3 = 0u128;

    // Find all set bits in the scalar
    let mut temp_scalar = scalar.clone();
    let mut bit_position = 0;

    while temp_scalar > BigInt::from(0) && bit_position < 254 {
        if &temp_scalar & BigInt::from(1) == BigInt::from(1) {
            // This bit is set, add the corresponding decomposition
            let (decomp_k0, decomp_k1, decomp_k2, decomp_k3, neg0, neg1, neg2, neg3) =
                POWER_OF_2_DECOMPOSITIONS[bit_position];

            if neg0 {
                k0 = k0.wrapping_sub(decomp_k0);
            } else {
                k0 = k0.wrapping_add(decomp_k0);
            }
            if neg1 {
                k1 = k1.wrapping_sub(decomp_k1);
            } else {
                k1 = k1.wrapping_add(decomp_k1);
            }
            if neg2 {
                k2 = k2.wrapping_sub(decomp_k2);
            } else {
                k2 = k2.wrapping_add(decomp_k2);
            }
            if neg3 {
                k3 = k3.wrapping_sub(decomp_k3);
            } else {
                k3 = k3.wrapping_add(decomp_k3);
            }
        }

        temp_scalar >>= 1;
        bit_position += 1;
    }

    // Handle signs by making coefficients positive and tracking negation flags
    let (final_k0, neg_flag0) = if (k0 as i128) < 0 {
        (k0.wrapping_neg(), true)
    } else {
        (k0, false)
    };

    let (final_k1, neg_flag1) = if (k1 as i128) < 0 {
        (k1.wrapping_neg(), true)
    } else {
        (k1, false)
    };

    let (final_k2, neg_flag2) = if (k2 as i128) < 0 {
        (k2.wrapping_neg(), true)
    } else {
        (k2, false)
    };

    let (final_k3, neg_flag3) = if (k3 as i128) < 0 {
        (k3.wrapping_neg(), true)
    } else {
        (k3, false)
    };

    (
        [final_k0, final_k1, final_k2, final_k3],
        [neg_flag0, neg_flag1, neg_flag2, neg_flag3],
    )
}

/// Verify that the decomposition is algebraically correct
///
/// Checks that k ≡ k0 + k1*λ + k2*λ² + k3*λ³ (mod r)
pub fn verify_decomposition(k: &Fr, coeffs: &[u128; 4], signs: &[bool; 4]) -> bool {
    let lambda_psi = get_bn254_frobenius_eigenvalue();

    let k0 = u128_to_fr(coeffs[0]);
    let k1 = u128_to_fr(coeffs[1]);
    let k2 = u128_to_fr(coeffs[2]);
    let k3 = u128_to_fr(coeffs[3]);

    let mut reconstructed = Fr::zero();
    if signs[0] {
        reconstructed -= k0;
    } else {
        reconstructed += k0;
    }
    if signs[1] {
        reconstructed -= k1 * lambda_psi;
    } else {
        reconstructed += k1 * lambda_psi;
    }
    if signs[2] {
        reconstructed -= k2 * lambda_psi * lambda_psi;
    } else {
        reconstructed += k2 * lambda_psi * lambda_psi;
    }
    if signs[3] {
        reconstructed -= k3 * lambda_psi * lambda_psi * lambda_psi;
    } else {
        reconstructed += k3 * lambda_psi * lambda_psi * lambda_psi;
    }

    *k == reconstructed
}

/// Get the maximum bit length of the decomposed coefficients
pub fn get_max_coefficient_bits(coeffs: &[u128; 4]) -> usize {
    coeffs
        .iter()
        .map(|k| (128 - k.leading_zeros()) as usize)
        .max()
        .unwrap_or(0)
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use ark_ff::UniformRand;
//     use ark_std::test_rng;

//     #[test]
//     fn test_decomposition_small_scalars() {
//         // Test k = 1
//         let k = Fr::from(1u64);
//         let k_bigint = fr_to_bigint(k);
//         let (coeffs, signs) = decompose_scalar_table_based(&k_bigint);

//         assert!(verify_decomposition(&k, &coeffs, &signs));

//         // Should be k0=1, k1=k2=k3=0, all positive
//         assert_eq!(coeffs[0], 1);
//         assert_eq!(coeffs[1], 0);
//         assert_eq!(coeffs[2], 0);
//         assert_eq!(coeffs[3], 0);
//         assert!(!signs[0]); // positive
//     }

//     #[test]
//     fn test_decomposition_random_scalars() {
//         let mut rng = test_rng();

//         for _ in 0..10 {
//             let k = Fr::rand(&mut rng);
//             let k_bigint = fr_to_bigint(k);
//             let (coeffs, signs) = decompose_scalar_table_based(&k_bigint);

//             // Verify algebraic correctness
//             assert!(verify_decomposition(&k, &coeffs, &signs));

//             // Check that coefficients are reasonably small (≤ 65 bits)
//             let max_bits = get_max_coefficient_bits(&coeffs);
//             assert!(max_bits <= 65, "Max bits: {}", max_bits);
//         }
//     }

//     #[test]
//     fn test_coefficient_size_reduction() {
//         let mut rng = test_rng();
//         let k = Fr::rand(&mut rng);
//         let k_bigint = fr_to_bigint(k);
//         let (coeffs, _signs) = decompose_scalar_table_based(&k_bigint);

//         let original_bits = k_bigint.bits();
//         let max_coefficient_bits = get_max_coefficient_bits(&coeffs);

//         // Should achieve significant reduction in bit length
//         assert!(max_coefficient_bits < original_bits);
//         assert!(max_coefficient_bits <= 65);
//     }

//     #[test]
//     fn test_conversion_functions() {
//         let mut rng = test_rng();

//         // Test round-trip conversion
//         for _ in 0..10 {
//             let original = Fr::rand(&mut rng);
//             let bigint = fr_to_bigint(original);
//             let converted_back = bigint_to_fr(&bigint);
//             assert_eq!(original, converted_back);
//         }

//         // Test u128 conversion
//         let val = 0x123456789abcdef0u128;
//         let fr_val = u128_to_fr(val);
//         let expected = Fr::from(val);
//         assert_eq!(fr_val, expected);
//     }
// }
