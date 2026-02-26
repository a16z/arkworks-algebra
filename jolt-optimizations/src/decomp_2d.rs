//! 2D GLV scalar multiplication implementations for BN254 G1

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
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

    ([k1_abs, k2_abs], [!k1_neg, !k2_neg])
}

/// Compute round(k * n / r) where n is 2 limbs.
/// Uses field mul for remainder, schoolbook for full product, exact division.
fn compute_quotient_2limb(
    k: &[u64; 4],
    n: [u64; 2],
    remainder_fr: Fr,
) -> u128 {
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
    let mut borrow = 0i128;
    for i in 0..4 {
        let d = (prod[i] as i128) - (rem[i] as i128) - borrow;
        diff[i] = d as u64;
        borrow = if d < 0 { 1 } else { 0 };
    }
    for i in 4..6 {
        let d = (prod[i] as i128) - borrow;
        diff[i] = d as u64;
        borrow = if d < 0 { 1 } else { 0 };
    }

    // Exact division by r: q = diff / r using Montgomery's method
    // q has at most 2 limbs
    let r_limbs = Fr::MODULUS.0;
    let q0 = diff[0].wrapping_mul(R_INV);
    // Subtract q0 * r from diff
    let mut sub_borrow = 0i128;
    for i in 0..4 {
        let p = (q0 as u128) * (r_limbs[i] as u128);
        let d = (diff[i] as i128) - (p as i128) - sub_borrow;
        diff[i] = d as u64;
        sub_borrow = ((p >> 64) as i128) - (d >> 64);
    }
    // diff[0] should now be 0, shift down
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
fn compute_quotient_1limb(
    k: &[u64; 4],
    n: u64,
    remainder_fr: Fr,
) -> u128 {
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
    let mut borrow = 0i128;
    for i in 0..4 {
        let d = (prod[i] as i128) - (rem[i] as i128) - borrow;
        diff[i] = d as u64;
        borrow = if d < 0 { 1 } else { 0 };
    }
    for i in 4..6 {
        let d = (prod[i] as i128) - borrow;
        diff[i] = d as u64;
        borrow = if d < 0 { 1 } else { 0 };
    }

    let r_limbs = Fr::MODULUS.0;
    let q0 = diff[0].wrapping_mul(R_INV);
    let mut sub_borrow = 0i128;
    for i in 0..4 {
        let p = (q0 as u128) * (r_limbs[i] as u128);
        let d = (diff[i] as i128) - (p as i128) - sub_borrow;
        diff[i] = d as u64;
        sub_borrow = ((p >> 64) as i128) - (d >> 64);
    }
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

/// Apply GLV endomorphism to G1 affine point: (x, y) → (β·x, y)
pub fn glv_endomorphism_affine(point: &G1Affine) -> G1Affine {
    G1Affine::new_unchecked(point.x * ENDO_COEFF, point.y)
}
