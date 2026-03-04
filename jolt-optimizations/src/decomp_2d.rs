//! 2D GLV scalar multiplication implementations for BN254 G1

use ark_bn254::{Fq, Fr, G1Projective};
use ark_ff::{BigInteger, MontFp, PrimeField};

/// GLV endomorphism coefficient for BN254 G1
const ENDO_COEFF: Fq =
    MontFp!("21888242871839275220042445260109153167277707414472061641714758635765020556616");

// Lattice basis coefficients (as Fr constants for field arithmetic)
const N11: Fr = MontFp!("147946756881789319000765030803803410728");
const N12: Fr = MontFp!("9931322734385697763");
const N21: Fr = MontFp!("9931322734385697763");
const N22: Fr = MontFp!("147946756881789319010696353538189108491");

// Raw limbs for schoolbook multiplication
const N22_LO: u64 = 0x0BE4E1541221250B;
const N22_HI: u64 = 0x6F4D8248EEB859FD;
const N12_VAL: u64 = 0x89D3256894D213E3;

// r^(-1) mod 2^64 for exact division
const R_INV: u64 = 0x3D1E0A6C10000001;

/// Zero-allocation 2D GLV scalar decomposition.
/// Uses field multiplication for remainder, exact division for quotient.
pub fn decompose_scalar_2d(scalar: Fr) -> ([<Fr as PrimeField>::BigInt; 2], [bool; 2]) {
    let k = scalar.into_bigint().0;

    // q1 = round(k * n22 / r)
    let q1 = compute_quotient_2limb(&k, [N22_LO, N22_HI], scalar * N22);
    // q2 = round(k * n12 / r)  (n12 sign handled below)
    let q2 = compute_quotient_1limb(&k, N12_VAL, scalar * N12);

    // Lattice vectors: u1=(N11, -N12) and u2=(N21, N22) where N21=N12
    // b = q1*u1 + q2*u2 → b1 = q1*N11 + q2*N12, b2 = -q1*N12 + q2*N22
    // k1 = scalar - b1, k2 = -b2
    let q1_fr = Fr::from(q1);
    let q2_fr = Fr::from(q2);
    let k1_fr = scalar - (q1_fr * N11 + q2_fr * N21);
    let k2_fr = q1_fr * N12 - q2_fr * N22;

    let k1_big = k1_fr.into_bigint();
    let k2_big = k2_fr.into_bigint();
    let half_r = Fr::MODULUS_MINUS_ONE_DIV_TWO;

    let k1_neg = k1_big > half_r;
    let k2_neg = k2_big > half_r;

    let k1_abs = if k1_neg {
        let mut tmp = Fr::MODULUS;
        tmp.sub_with_borrow(&k1_big);
        tmp
    } else {
        k1_big
    };
    let k2_abs = if k2_neg {
        let mut tmp = Fr::MODULUS;
        tmp.sub_with_borrow(&k2_big);
        tmp
    } else {
        k2_big
    };

    debug_assert!(
        k1_abs.num_bits() <= 130,
        "k1_abs too large: {} bits",
        k1_abs.num_bits()
    );
    debug_assert!(
        k2_abs.num_bits() <= 130,
        "k2_abs too large: {} bits",
        k2_abs.num_bits()
    );

    ([k1_abs, k2_abs], [!k1_neg, !k2_neg])
}

/// a - b - borrow_in → (result, borrow_out), pure unsigned arithmetic
#[inline(always)]
fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let sub = (b as u128) + (borrow as u128);
    let a128 = a as u128;
    (a128.wrapping_sub(sub) as u64, (a128 < sub) as u64)
}

/// Subtract q_limb * r from diff (multiply-and-subtract), pure u64 arithmetic
#[inline(always)]
fn sub_mul_limb(diff: &mut [u64; 6], q_limb: u64, r_limbs: &[u64; 4]) {
    let mut carry = 0u64;
    let mut borrow = 0u64;
    for i in 0..4 {
        let p = (q_limb as u128) * (r_limbs[i] as u128) + (carry as u128);
        let (val, b) = sbb(diff[i], p as u64, borrow);
        diff[i] = val;
        carry = (p >> 64) as u64;
        borrow = b;
    }
    for i in 4..6 {
        let (val, b) = sbb(diff[i], carry, borrow);
        diff[i] = val;
        carry = 0;
        borrow = b;
    }
}

/// Compute round(k * n / r) where n is 2 limbs.
/// Uses field mul for remainder, schoolbook for full product, exact division.
fn compute_quotient_2limb(k: &[u64; 4], n: [u64; 2], remainder_fr: Fr) -> u128 {
    // Full product k * n (6 limbs)
    let mut prod = [0u64; 6];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..2 {
            let p = (k[i] as u128) * (n[j] as u128) + (prod[i + j] as u128) + carry;
            prod[i + j] = p as u64;
            carry = p >> 64;
        }
        prod[i + 2] = carry as u64;
    }

    let rem = remainder_fr.into_bigint().0;

    // diff = prod - rem (6-limb - 4-limb), result is q * r
    let mut diff = [0u64; 6];
    let mut borrow = 0u64;
    for i in 0..4 {
        let (val, b) = sbb(prod[i], rem[i], borrow);
        diff[i] = val;
        borrow = b;
    }
    for i in 4..6 {
        let (val, b) = sbb(prod[i], 0, borrow);
        diff[i] = val;
        borrow = b;
    }

    // Exact division by r: q = diff / r using Montgomery's method
    // q has at most 2 limbs
    let r_limbs = Fr::MODULUS.0;
    let q0 = diff[0].wrapping_mul(R_INV);
    sub_mul_limb(&mut diff, q0, &r_limbs);
    debug_assert!(
        diff[0] == 0,
        "2limb: Montgomery step failed: diff[0]={:016X}",
        diff[0]
    );
    let q1 = diff[1].wrapping_mul(R_INV);

    let q = (q0 as u128) | ((q1 as u128) << 64);

    // Rounding: if 2 * remainder > r, increment quotient
    let two_rem_gt_r = bigint4_gt_half_r(&rem);
    if two_rem_gt_r {
        q + 1
    } else {
        q
    }
}

/// Compute round(k * n / r) where n is 1 limb.
fn compute_quotient_1limb(k: &[u64; 4], n: u64, remainder_fr: Fr) -> u128 {
    // Full product k * n (5 limbs)
    let mut prod = [0u64; 6];
    let mut carry = 0u128;
    for i in 0..4 {
        let p = (k[i] as u128) * (n as u128) + carry;
        prod[i] = p as u64;
        carry = p >> 64;
    }
    prod[4] = carry as u64;

    let rem = remainder_fr.into_bigint().0;

    let mut diff = [0u64; 6];
    let mut borrow = 0u64;
    for i in 0..4 {
        let (val, b) = sbb(prod[i], rem[i], borrow);
        diff[i] = val;
        borrow = b;
    }
    for i in 4..6 {
        let (val, b) = sbb(prod[i], 0, borrow);
        diff[i] = val;
        borrow = b;
    }

    let r_limbs = Fr::MODULUS.0;
    let q0 = diff[0].wrapping_mul(R_INV);
    sub_mul_limb(&mut diff, q0, &r_limbs);
    debug_assert!(
        diff[0] == 0,
        "1limb: Montgomery step failed: diff[0]={:016X}",
        diff[0]
    );
    let q1 = diff[1].wrapping_mul(R_INV);

    let q = (q0 as u128) | ((q1 as u128) << 64);

    let two_rem_gt_r = bigint4_gt_half_r(&rem);
    if two_rem_gt_r {
        q + 1
    } else {
        q
    }
}

#[inline]
fn bigint4_gt_half_r(val: &[u64; 4]) -> bool {
    // half_r = (r - 1) / 2
    const HALF_R: [u64; 4] = [
        0xA1F0FAC9F8000000,
        0x9419F4243CDCB848,
        0xDC2822DB40C0AC2E,
        0x183227397098D014,
    ];
    for i in (0..4).rev() {
        if val[i] > HALF_R[i] {
            return true;
        }
        if val[i] < HALF_R[i] {
            return false;
        }
    }
    false
}

/// Apply GLV endomorphism to G1 projective point: (X, Y, Z) → (β·X, Y, Z)
pub fn glv_endomorphism(point: &G1Projective) -> G1Projective {
    let mut res = *point;
    res.x *= ENDO_COEFF;
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::G1Affine;
    use ark_ec::{CurveGroup, PrimeGroup};
    use ark_ff::{MontFp, PrimeField};
    use ark_std::UniformRand;
    use std::ops::Mul;

    const LAMBDA: Fr =
        MontFp!("21888242871839275217838484774961031246154997185409878258781734729429964517155");

    fn is_on_curve_affine(p: &G1Affine) -> bool {
        if p.infinity {
            return true;
        }
        p.y * p.y == p.x * p.x * p.x + Fq::from(3u64)
    }

    fn is_on_curve(p: &G1Projective) -> bool {
        is_on_curve_affine(&p.into_affine())
    }

    #[test]
    fn test_decomp_identity_small_scalars() {
        let lambda = LAMBDA;

        for val in [0u64, 1, 2, 42, 123456789, u64::MAX] {
            let scalar = Fr::from(val);
            let (coeffs, signs) = decompose_scalar_2d(scalar);

            let k1 = Fr::from_bigint(coeffs[0]).unwrap();
            let k2 = Fr::from_bigint(coeffs[1]).unwrap();
            let signed_k1 = if signs[0] { k1 } else { -k1 };
            let signed_k2 = if signs[1] { k2 } else { -k2 };
            let reconstructed = signed_k1 + lambda * signed_k2;
            assert_eq!(reconstructed, scalar, "identity failed for val={val}");
        }
    }

    #[test]
    fn test_decomp_identity_random() {
        let lambda = LAMBDA;

        let mut rng = ark_std::test_rng();
        for _ in 0..1000 {
            let scalar = Fr::rand(&mut rng);
            let (coeffs, signs) = decompose_scalar_2d(scalar);

            let k1 = Fr::from_bigint(coeffs[0]).unwrap();
            let k2 = Fr::from_bigint(coeffs[1]).unwrap();
            let signed_k1 = if signs[0] { k1 } else { -k1 };
            let signed_k2 = if signs[1] { k2 } else { -k2 };
            let reconstructed = signed_k1 + lambda * signed_k2;
            assert_eq!(reconstructed, scalar, "identity failed for random scalar");
        }
    }

    #[test]
    fn test_glv_endomorphism_on_curve() {
        let mut rng = ark_std::test_rng();
        for _ in 0..100 {
            let p = G1Projective::rand(&mut rng);
            let endo_proj = glv_endomorphism(&p);
            assert!(
                is_on_curve(&endo_proj),
                "proj endomorphism produced off-curve point"
            );
        }
    }

    #[test]
    fn test_glv_mul_matches_standard() {
        use crate::glv_two::shamir_glv_mul_2d;

        let gen_proj = G1Projective::generator();
        let mut rng = ark_std::test_rng();

        for _ in 0..100 {
            let scalar = Fr::rand(&mut rng);
            let expected = gen_proj.mul(scalar);
            assert!(is_on_curve(&expected), "standard mul off-curve");

            let (coeffs, signs) = decompose_scalar_2d(scalar);
            let glv_proj = glv_endomorphism(&gen_proj);
            let result = shamir_glv_mul_2d(&[gen_proj, glv_proj], &coeffs, &signs);
            assert!(is_on_curve(&result), "shamir result off-curve");

            assert_eq!(
                expected.into_affine(),
                result.into_affine(),
                "GLV result doesn't match standard mul"
            );
        }
    }

    #[test]
    fn test_glv_mul_random_base() {
        use crate::glv_two::shamir_glv_mul_2d;

        let mut rng = ark_std::test_rng();

        for _ in 0..100 {
            let base = G1Projective::rand(&mut rng);
            let scalar = Fr::rand(&mut rng);

            let expected = base.mul(scalar);
            let (coeffs, signs) = decompose_scalar_2d(scalar);
            let glv_proj = glv_endomorphism(&base);
            let result = shamir_glv_mul_2d(&[base, glv_proj], &coeffs, &signs);

            assert!(is_on_curve(&result), "random base GLV result off-curve");
            assert_eq!(
                expected.into_affine(),
                result.into_affine(),
                "random base GLV doesn't match standard"
            );
        }
    }

    #[test]
    fn test_vector_add_scalar_mul_g1_online() {
        use crate::dory_g1::vector_add_scalar_mul_g1_online;

        let mut rng = ark_std::test_rng();
        let n = 128;

        let generators: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let mut v: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let v_orig = v.clone();
        let scalar = Fr::rand(&mut rng);

        vector_add_scalar_mul_g1_online(&mut v, &generators, scalar);

        for i in 0..n {
            assert!(
                is_on_curve(&v[i]),
                "v[{i}] off-curve after vector_add_scalar_mul"
            );
            let expected = v_orig[i] + generators[i].mul(scalar);
            assert_eq!(
                v[i].into_affine(),
                expected.into_affine(),
                "v[{i}] doesn't match expected"
            );
        }
    }

    #[test]
    fn test_vector_scalar_mul_add_gamma_g1_online() {
        use crate::dory_g1::vector_scalar_mul_add_gamma_g1_online;

        let mut rng = ark_std::test_rng();
        let n = 128;

        let gamma: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let mut v: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let v_orig = v.clone();
        let scalar = Fr::rand(&mut rng);

        vector_scalar_mul_add_gamma_g1_online(&mut v, scalar, &gamma);

        for i in 0..n {
            assert!(
                is_on_curve(&v[i]),
                "v[{i}] off-curve after vector_scalar_mul_add_gamma"
            );
            let expected = v_orig[i].mul(scalar) + gamma[i];
            assert_eq!(
                v[i].into_affine(),
                expected.into_affine(),
                "v[{i}] doesn't match expected"
            );
        }
    }
}
