//! 4D GLV scalar multiplication for BN254 G2
//! Three methods: (1) online, (2) precomputed full, (3) signed table

use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::{AdditiveGroup, CurveGroup};
use ark_ff::{BigInteger, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use rayon::prelude::*;
use std::cell::RefCell;

use crate::decomp_4d::decompose_scalar_4d;
use crate::frobenius::frobenius_psi_power_projective;

/// Minimum collection length to justify rayon par_iter overhead
const MIN_PAR_SIZE: usize = 64;

struct CachedAffineBases {
    point: G2Projective,
    bases: [G2Affine; 4],
}

/// Bitwise comparison of projective coordinates (not mathematical equality).
/// Detects whether the *same projective representation* was passed again,
/// which is the caller's pattern (same base_proj every call).
#[inline]
fn same_proj_repr(a: &G2Projective, b: &G2Projective) -> bool {
    a.x == b.x && a.y == b.y && a.z == b.z
}

thread_local! {
    static FROBENIUS_CACHE: RefCell<Option<CachedAffineBases>> = const { RefCell::new(None) };
}

fn get_or_compute_affine_bases(point: &G2Projective) -> [G2Affine; 4] {
    FROBENIUS_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(ref cached) = *cache {
            if same_proj_repr(&cached.point, point) {
                return cached.bases;
            }
        }

        let proj_bases = [
            *point,
            frobenius_psi_power_projective(point, 1),
            frobenius_psi_power_projective(point, 2),
            frobenius_psi_power_projective(point, 3),
        ];
        let affine_vec = G2Projective::normalize_batch(&proj_bases);
        let bases = [affine_vec[0], affine_vec[1], affine_vec[2], affine_vec[3]];

        *cache = Some(CachedAffineBases {
            point: *point,
            bases,
        });
        bases
    })
}

/// Online 4D GLV scalar multiplication
pub fn glv_four_scalar_mul_online(scalar: Fr, points: &[G2Projective]) -> Vec<G2Projective> {
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    if points.len() == 1 {
        let bases = get_or_compute_affine_bases(&points[0]);
        return vec![shamir_glv_mul_4d_affine(&bases, &coeffs, &signs)];
    }

    let body = |point: &G2Projective| {
        let bases = get_or_compute_affine_bases(point);
        shamir_glv_mul_4d_affine(&bases, &coeffs, &signs)
    };

    if points.len() >= MIN_PAR_SIZE {
        points.par_iter().map(body).collect()
    } else {
        points.iter().map(body).collect()
    }
}

/// Shamir's trick for 4-point scalar mul (affine bases, mixed addition).
/// Pre-applies signs to eliminate inner-loop branch.
pub(crate) fn shamir_glv_mul_4d_affine(
    bases: &[G2Affine; 4],
    coeffs: &[<Fr as PrimeField>::BigInt; 4],
    signs: &[bool; 4],
) -> G2Projective {
    let effective: [G2Affine; 4] = std::array::from_fn(|i| {
        if signs[i] { -bases[i] } else { bases[i] }
    });

    let max_bits = coeffs
        .iter()
        .map(|c| c.num_bits() as usize)
        .max()
        .unwrap_or(0);

    let mut result = G2Projective::zero();
    for bit_idx in (0..max_bits).rev() {
        result.double_in_place();

        for i in 0..4 {
            if coeffs[i].get_bit(bit_idx) {
                result += effective[i];
            }
        }
    }

    result
}

/// Precomputed data for 4D GLV with Shamir table
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrecomputedShamir4Data {
    pub shamir_tables: Vec<PrecomputedShamir4Table>,
}

/// Shamir lookup table: all 256 combinations for [P, ψ(P), ψ²(P), ψ³(P)] with signs
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrecomputedShamir4Table {
    pub table: Vec<G2Projective>, // 2^4 points × 2^4 signs (256 elements)
}

impl PrecomputedShamir4Table {
    /// Create table for [P, ψ(P), ψ²(P), ψ³(P)] with all sign combinations
    pub fn new(bases: &[G2Projective; 4]) -> Self {
        let mut table = vec![G2Projective::zero(); 256];

        table.iter_mut().enumerate().for_each(|(idx, point)| {
            let point_mask = idx & 0xF;
            let sign_mask = idx >> 4;

            *point = G2Projective::zero();
            for i in 0..4 {
                if (point_mask >> i) & 1 == 1 {
                    if (sign_mask >> i) & 1 == 1 {
                        *point -= bases[i];
                    } else {
                        *point += bases[i];
                    }
                }
            }
        });

        Self { table }
    }

    #[inline]
    pub fn get(&self, point_mask: usize, sign_mask: usize) -> G2Projective {
        self.table[point_mask | (sign_mask << 4)]
    }
}

impl PrecomputedShamir4Data {
    pub fn new(points: &[G2Projective]) -> Self {
        let body = |point: &G2Projective| {
            let frobenius_bases = [
                *point,
                frobenius_psi_power_projective(point, 1),
                frobenius_psi_power_projective(point, 2),
                frobenius_psi_power_projective(point, 3),
            ];
            PrecomputedShamir4Table::new(&frobenius_bases)
        };

        let shamir_tables = if points.len() >= MIN_PAR_SIZE {
            points.par_iter().map(body).collect()
        } else {
            points.iter().map(body).collect()
        };

        Self { shamir_tables }
    }
}

/// Precompute for multiple points
pub fn glv_four_precompute(points: &[G2Projective]) -> PrecomputedShamir4Data {
    PrecomputedShamir4Data::new(points)
}

/// Scalar multiplication using precomputed data
pub fn glv_four_scalar_mul(data: &PrecomputedShamir4Data, scalar: Fr) -> Vec<G2Projective> {
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    let body = |table: &PrecomputedShamir4Table| {
        shamir_glv_mul_4d_precomputed(table, &coeffs, &signs)
    };

    if data.shamir_tables.len() >= MIN_PAR_SIZE {
        data.shamir_tables.par_iter().map(body).collect()
    } else {
        data.shamir_tables.iter().map(body).collect()
    }
}

/// Shamir's trick using precomputed table
pub(crate) fn shamir_glv_mul_4d_precomputed(
    table: &PrecomputedShamir4Table,
    coeffs: &[<Fr as PrimeField>::BigInt; 4],
    signs: &[bool; 4],
) -> G2Projective {
    let mut result = G2Projective::zero();
    let max_bits = coeffs
        .iter()
        .map(|c| c.num_bits() as usize)
        .max()
        .unwrap_or(0);

    for bit_idx in (0..max_bits).rev() {
        result = result.double();

        // Build masks for table lookup
        let mut point_mask = 0;
        let mut sign_mask = 0;

        for i in 0..4 {
            if coeffs[i].get_bit(bit_idx) {
                point_mask |= 1 << i;
                if signs[i] {
                    // signs[i] = true means negative in 4D decomposition
                    sign_mask |= 1 << i;
                }
            }
        }

        if point_mask != 0 {
            result += table.get(point_mask, sign_mask);
        }
    }

    result
}

/// Precomputed data for 2-bit windowed signed method
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Windowed2Signed4Data {
    pub windowed2_tables: Vec<Windowed2Signed4Table>,
}

/// 2-bit signed table: stores [±P, ±2P, ±3P] for each base
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Windowed2Signed4Table {
    pub signed_multiples: Vec<G2Projective>, // 4 bases × 6 variants (24 elements)
}

impl Windowed2Signed4Table {
    /// Create signed multiples for 2-bit processing
    pub fn new(bases: &[G2Projective; 4]) -> Self {
        let mut signed_multiples = vec![G2Projective::zero(); 24];

        for (base_idx, &base) in bases.iter().enumerate() {
            let offset = base_idx * 6;
            signed_multiples[offset] = base; // 1*base
            signed_multiples[offset + 1] = base.double(); // 2*base
            signed_multiples[offset + 2] = base + base.double(); // 3*base
            signed_multiples[offset + 3] = -base; // -1*base
            signed_multiples[offset + 4] = -base.double(); // -2*base
            signed_multiples[offset + 5] = -(base + base.double()); // -3*base
        }

        Self { signed_multiples }
    }

    /// Get 2-bit windowed combination
    #[inline]
    pub fn get_windowed2(&self, coeffs: &[i8; 4]) -> G2Projective {
        let mut result = G2Projective::zero();

        for (i, &coeff) in coeffs.iter().enumerate() {
            if coeff != 0 {
                let abs_coeff = coeff.abs() as usize;
                if abs_coeff <= 3 {
                    let idx = i * 6 + abs_coeff - 1 + if coeff < 0 { 3 } else { 0 };
                    result += self.signed_multiples[idx];
                }
            }
        }

        result
    }
}

impl Windowed2Signed4Data {
    pub fn new(points: &[G2Projective]) -> Self {
        let body = |point: &G2Projective| {
            let frobenius_bases = [
                *point,
                frobenius_psi_power_projective(point, 1),
                frobenius_psi_power_projective(point, 2),
                frobenius_psi_power_projective(point, 3),
            ];
            Windowed2Signed4Table::new(&frobenius_bases)
        };

        let windowed2_tables = if points.len() >= MIN_PAR_SIZE {
            points.par_iter().map(body).collect()
        } else {
            points.iter().map(body).collect()
        };

        Self { windowed2_tables }
    }
}

/// Precompute for 2-bit windowed signed method
pub fn glv_four_precompute_windowed2_signed(points: &[G2Projective]) -> Windowed2Signed4Data {
    Windowed2Signed4Data::new(points)
}

/// Scalar multiplication using 2-bit windowed signed method
pub fn glv_four_scalar_mul_windowed2_signed(
    data: &Windowed2Signed4Data,
    scalar: Fr,
) -> Vec<G2Projective> {
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    let body = |table: &Windowed2Signed4Table| {
        glv_four_scalar_mul_windowed2_signed_single(table, &coeffs, &signs)
    };

    if data.windowed2_tables.len() >= MIN_PAR_SIZE {
        data.windowed2_tables.par_iter().map(body).collect()
    } else {
        data.windowed2_tables.iter().map(body).collect()
    }
}

/// 2-bit windowed signed multiplication for single point
fn glv_four_scalar_mul_windowed2_signed_single(
    table: &Windowed2Signed4Table,
    coeffs: &[<Fr as PrimeField>::BigInt; 4],
    signs: &[bool; 4],
) -> G2Projective {
    // Convert scalars to signed coefficients for 2-bit windowed processing
    let scalar_coeffs: Vec<Vec<i8>> = coeffs
        .iter()
        .zip(signs.iter())
        .map(|(scalar, &is_negative)| {
            let mut coeffs = Vec::new();
            let scalar_ref = scalar.as_ref();

            for limb in scalar_ref {
                for window_idx in 0..32 {
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
    let mut result = G2Projective::zero();

    for coeff_idx in (0..max_coeffs).rev() {
        // Quadruple (process 2 bits)
        result = result.double().double();

        // Extract coefficients for this window
        let mut window_coeffs = [0i8; 4];
        for i in 0..4 {
            if coeff_idx < scalar_coeffs[i].len() {
                window_coeffs[i] = scalar_coeffs[i][coeff_idx];
            }
        }

        // Use signed lookup if any coefficient is non-zero
        if window_coeffs.iter().any(|&c| c != 0) {
            result += table.get_windowed2(&window_coeffs);
        }
    }

    result
}

/// Scalar multiplication using precomputed data and decomposed scalar
pub fn glv_four_scalar_mul_decomposed(
    data: &PrecomputedShamir4Data,
    coeffs: &[<Fr as PrimeField>::BigInt; 4],
    signs: &[bool; 4],
) -> Vec<G2Projective> {
    let body = |table: &PrecomputedShamir4Table| {
        shamir_glv_mul_4d_precomputed(table, coeffs, signs)
    };

    if data.shamir_tables.len() >= MIN_PAR_SIZE {
        data.shamir_tables.par_iter().map(body).collect()
    } else {
        data.shamir_tables.iter().map(body).collect()
    }
}
