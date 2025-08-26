//! Direct polynomial quotient computation for Schwartz-Zippel verification
//!
//! This module provides functionality to compute quotient polynomials directly in F[x]
//! rather than computing in Fp12 first. This is necessary for correctly handling
//! arbitrary exponents in the batched expression verification scheme.

use ark_bn254::Fq;
use ark_ff::{BigInteger, One, PrimeField, Zero};

// === Helpers for quotient-carrying arithmetic ===

#[inline(always)]
fn c18() -> Fq {
    Fq::from(18u64)
}
#[inline(always)]
fn c82() -> Fq {
    Fq::from(82u64)
}

/// Polynomial division by a monic polynomial
/// Returns (quotient, remainder) where dividend = quotient * divisor + remainder
fn poly_div_rem_monic(mut dividend: Vec<Fq>, divisor: &[Fq]) -> (Vec<Fq>, Vec<Fq>) {
    assert!(!divisor.is_empty(), "divisor must be non-empty");
    assert!(
        divisor.last().unwrap().is_one(),
        "divisor must be monic (leading coefficient = 1)"
    );

    // Remove leading zeros from dividend
    while dividend.len() > 1 && dividend.last().unwrap().is_zero() {
        dividend.pop();
    }

    let divisor_deg = divisor.len() - 1;
    if dividend.len() <= divisor_deg {
        return (vec![Fq::zero()], dividend);
    }

    let mut quotient = vec![Fq::zero(); dividend.len() - divisor_deg];

    // Long division
    for i in (0..quotient.len()).rev() {
        let coeff = dividend[i + divisor_deg];
        quotient[i] = coeff;

        // Subtract coeff * divisor from dividend
        for j in 0..divisor.len() {
            dividend[i + j] -= coeff * divisor[j];
        }
    }

    // Remainder is the lower degree terms
    dividend.truncate(divisor_deg);

    // Remove leading zeros from remainder
    while dividend.len() > 1 && dividend.last().unwrap().is_zero() {
        dividend.pop();
    }

    (quotient, dividend)
}

/// Multiply two deg-≤11 polys in Fp[X]. Reduce mod g, and return:
/// rem: deg ≤ 11 representative, q: quotient s.t. a*b = rem + g*q.
fn mul_mod_g_with_quot(a: &[Fq; 12], b: &[Fq; 12]) -> ([Fq; 12], Vec<Fq>) {
    // 12x12 convolution up to deg 22
    let mut t = vec![Fq::zero(); 23];
    for i in 0..12 {
        if a[i].is_zero() {
            continue;
        }
        for j in 0..12 {
            if b[j].is_zero() {
                continue;
            }
            t[i + j] += a[i] * b[j];
        }
    }
    // fold using X^12 = 18 X^6 - 82, track quotient coeff at X^{k-12}
    let mut q = vec![Fq::zero(); 11];
    let k18 = c18();
    let k82 = c82();
    for k in (12..=22).rev() {
        let c = t[k];
        if c.is_zero() {
            continue;
        }
        t[k] = Fq::zero();
        q[k - 12] += c; // take out c*X^k as g * (c*X^{k-12}) + lower
        t[k - 6] += k18 * c;
        t[k - 12] -= k82 * c;
    }
    let mut rem = [Fq::zero(); 12];
    rem.copy_from_slice(&t[0..12]);
    while q.last().map_or(false, |x| x.is_zero()) {
        q.pop();
    }
    (rem, q)
}

/// (R,Q) ← (R,Q) * base, where base has deg ≤ 11.
/// Keeps R reduced (deg ≤ 11) and updates Q in the *unreduced* ring.
fn mul_pair_by_base(rem: &mut [Fq; 12], quot: &mut Vec<Fq>, base: &[Fq; 12]) {
    let (new_rem, q_rb) = mul_mod_g_with_quot(rem, base);

    // quot * base (plain convolution)
    let mut qb = vec![Fq::zero(); quot.len() + 11];
    for i in 0..quot.len() {
        if quot[i].is_zero() {
            continue;
        }
        for j in 0..12 {
            if base[j].is_zero() {
                continue;
            }
            qb[i + j] += quot[i] * base[j];
        }
    }

    // Q' = q_rb + qb
    let max_len = core::cmp::max(q_rb.len(), qb.len());
    let mut new_q = vec![Fq::zero(); max_len];
    for i in 0..q_rb.len() {
        new_q[i] += q_rb[i];
    }
    for i in 0..qb.len() {
        new_q[i] += qb[i];
    }
    while new_q.last().map_or(false, |x| x.is_zero()) {
        new_q.pop();
    }

    *rem = new_rem;
    *quot = new_q;
}

/// Reduce arbitrary poly mod g to deg ≤ 11 rep (no quotient needed here).
fn reduce_mod_g(mut p: Vec<Fq>, g: &[Fq]) -> [Fq; 12] {
    // long division by monic g
    let (_q, mut r) = poly_div_rem_monic(p, g);
    r.resize(12, Fq::zero());
    let mut out = [Fq::zero(); 12];
    let upto = core::cmp::min(12, r.len());
    out[..upto].copy_from_slice(&r[..upto]);
    out
}

/// Multiply running pair by base^exp using square-and-multiply,
/// calling mul_pair_by_base whenever the bit is set.
fn mul_pair_by_pow(rem: &mut [Fq; 12], quot: &mut Vec<Fq>, base_raw: &[Fq], exp: &Fq, g: &[Fq]) {
    if exp.is_zero() {
        return;
    }
    // base power representative (deg ≤ 11)
    let mut pow_rem = reduce_mod_g(base_raw.to_vec(), g);

    let e = exp.into_bigint();
    let nbits = Fq::MODULUS_BIT_SIZE as usize;
    for i in 0..nbits {
        if e.get_bit(i) {
            mul_pair_by_base(rem, quot, &pow_rem);
        }
        // square pow_rem (we only need rem part here)
        let (sq_rem, _q_drop) = mul_mod_g_with_quot(&pow_rem, &pow_rem);
        pow_rem = sq_rem;
    }
}

/// Main function to compute quotient polynomial for expression verification
///
/// Given:
/// - lhs: z_i'(X) polynomial
/// - rhs_terms: [(z_{i,j}(X), e_{i,j})] pairs
/// - g: the polynomial g(X) = X^12 - 18X^6 + 82
///
/// Computes quotient q_i(X) such that:
/// z_i'(X) - ∏_j z_{i,j}(X)^{e_{i,j}} = q_i(X) * g(X)
///
/// Returns the quotient polynomial q_i(X)
pub fn compute_quotient_direct(
    lhs: &[Fq],
    rhs_terms: &[(Vec<Fq>, Fq)],
    g: &[Fq],
) -> Result<Vec<Fq>, String> {
    // 1) Start with (R,Q) = (1, 0)
    let mut R = [Fq::zero(); 12];
    R[0] = Fq::one();
    let mut Q: Vec<Fq> = vec![];

    // 2) For each term, multiply by base^exp using the pair method
    for (poly, exp) in rhs_terms {
        mul_pair_by_pow(&mut R, &mut Q, poly, exp, g);
    }

    // 3) For honest inputs, lhs ≡ R (mod g). The correct expression quotient is q = -Q.
    //    You can skip division entirely; we still sanity-check in debug.
    #[cfg(debug_assertions)]
    {
        let lhs_rem = reduce_mod_g(lhs.to_vec(), g);
        debug_assert!(
            (0..12).all(|i| lhs_rem[i] == R[i]),
            "lhs != product (mod g)"
        );
    }

    // q = -Q
    for c in Q.iter_mut() {
        *c = -*c;
    }
    while Q.last().map_or(false, |x| x.is_zero()) {
        Q.pop();
    }

    Ok(Q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    fn g_coeffs() -> Vec<Fq> {
        let mut g = vec![Fq::zero(); 13];
        g[0] = Fq::from(82u64);
        g[6] = -Fq::from(18u64);
        g[12] = Fq::one();
        g
    }

    #[test]
    fn test_mul_mod_g_with_quot() {
        // Test: (X)(X) = X^2 mod g
        let mut a = [Fq::zero(); 12];
        let mut b = [Fq::zero(); 12];
        a[1] = Fq::one(); // X
        b[1] = Fq::one(); // X

        let (rem, q) = mul_mod_g_with_quot(&a, &b);

        // X * X = X^2, which is degree < 12, so quotient should be zero
        assert!(q.is_empty() || q.iter().all(|x| x.is_zero()));
        assert!(rem[2].is_one()); // coefficient of X^2
    }

    #[test]
    fn test_compute_quotient_simple_protocol_semantics() {
        let g = g_coeffs();

        // q(X) arbitrary; lhs_unreduced = q*g + 1, but the prover only sees lhs_reduced = 1
        let q = vec![Fq::from(5u64), Fq::from(3u64), Fq::one()];

        // Build unreduced lhs = q*g + 1, then reduce to degree ≤ 11 (which is just 1)
        let mut lhs_unreduced = vec![Fq::zero(); q.len() + g.len() - 1];
        for i in 0..q.len() {
            for j in 0..g.len() {
                lhs_unreduced[i + j] += q[i] * g[j];
            }
        }
        lhs_unreduced[0] += Fq::one();
        let lhs_reduced = poly_div_rem_monic(lhs_unreduced.clone(), &g).1; // = [1]

        // RHS = 1
        let rhs_terms = vec![(vec![Fq::one()], Fq::one())];

        // Under protocol semantics (all inputs reduced), quotient must be 0
        let computed_q = compute_quotient_direct(&lhs_reduced, &rhs_terms, &g).unwrap();
        assert!(computed_q.iter().all(|c| c.is_zero()));
    }

    #[test]
    fn test_compute_quotient_with_powers() {
        let g = g_coeffs();

        // Test: (X^2 + 1)^2
        let poly = vec![Fq::one(), Fq::zero(), Fq::one()]; // X^2 + 1
        let exp = Fq::from(2u64);

        // Compute (X^2 + 1)^2 = X^4 + 2X^2 + 1 and reduce mod g
        let mut lhs_unreduced = vec![Fq::zero(); 5];
        lhs_unreduced[0] = Fq::one(); // 1
        lhs_unreduced[2] = Fq::from(2u64); // 2X^2
        lhs_unreduced[4] = Fq::one(); // X^4

        let (_, lhs_rem) = poly_div_rem_monic(lhs_unreduced, &g);
        let mut lhs = vec![Fq::zero(); 12];
        for i in 0..lhs_rem.len().min(12) {
            lhs[i] = lhs_rem[i];
        }

        // RHS is the original polynomial with exponent 2
        let rhs_terms = vec![(poly.clone(), exp)];

        // Compute quotient
        let quotient = compute_quotient_direct(&lhs, &rhs_terms, &g).unwrap();

        // The quotient should be zero since (X^2+1)^2 has degree 4 < 12
        assert!(quotient.is_empty() || quotient.iter().all(|x| x.is_zero()));
    }

    #[test]
    fn test_large_exponent() {
        let mut rng = test_rng();
        let g = g_coeffs();

        // Create a random polynomial of degree < 12
        let mut poly = vec![Fq::zero(); 8];
        for i in 0..8 {
            poly[i] = Fq::rand(&mut rng);
        }

        // Large exponent - this will require reduction
        let exp = Fq::from(17u64);

        // For this test, we compute poly^17 mod g manually to get LHS
        // This is complex, so we'll just verify the quotient computation works

        // We'll use a simpler approach: use X^13 which definitely needs reduction
        let x13_poly = {
            let mut p = vec![Fq::zero(); 14];
            p[13] = Fq::one(); // X^13
            p
        };

        // X^13 mod g = X^13 mod (X^12 - 18X^6 + 82) = X * (18X^6 - 82) = 18X^7 - 82X
        let lhs = {
            let mut l = vec![Fq::zero(); 12];
            l[1] = -Fq::from(82u64); // -82X
            l[7] = Fq::from(18u64); // 18X^7
            l
        };

        // RHS = X^13
        let x_poly = vec![Fq::zero(), Fq::one()]; // X
        let rhs_terms = vec![(x_poly, Fq::from(13u64))];

        // Compute quotient
        let quotient = compute_quotient_direct(&lhs, &rhs_terms, &g).unwrap();

        // The quotient should be [0, -1] representing -X (since X^13 = g*X + (18X^7 - 82X))
        assert_eq!(quotient.len(), 2);
        assert!(quotient[0].is_zero());
        assert_eq!(quotient[1], -Fq::one());
    }

    #[test]
    fn test_multiple_terms_product() {
        let g = g_coeffs();

        // Simple test: (X) * (X^2) = X^3
        let x = vec![Fq::zero(), Fq::one()]; // X
        let x2 = vec![Fq::zero(), Fq::zero(), Fq::one()]; // X^2

        // LHS = X^3
        let lhs = vec![Fq::zero(), Fq::zero(), Fq::zero(), Fq::one()];

        // RHS = X * X^2
        let rhs_terms = vec![(x.clone(), Fq::one()), (x2.clone(), Fq::one())];

        // Compute quotient
        let quotient = compute_quotient_direct(&lhs, &rhs_terms, &g).unwrap();

        // X^3 has degree < 12, so quotient should be zero
        assert!(quotient.is_empty() || quotient.iter().all(|x| x.is_zero()));
    }
}
