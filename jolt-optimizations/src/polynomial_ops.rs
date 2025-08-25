//! Polynomial operations over Fq for BN254
//!
//! This module provides polynomial arithmetic operations specifically for
//! working with polynomials over the BN254 base field Fq.

use ark_bn254::Fq;
use ark_ff::{Field, One, Zero};

/// Evaluate g(X) = X^12 - 18 X^6 + 82 at a given point r.
///
/// This is the minimal polynomial for the BN254 Fq12 extension field
/// when viewed as Fq[X]/(g(X)).
///
/// # Arguments
/// * `r` - The point at which to evaluate g(X)
///
/// # Returns
/// The value g(r) in Fq
pub fn g_eval(r: &Fq) -> Fq {
    let r2 = *r * r;       // r^2
    let r3 = r2 * r;       // r^3
    let r6 = r3.square();  // r^6
    let r12 = r6.square(); // r^12
    r12 - (Fq::from(18u64) * r6) + Fq::from(82u64)
}

/// Horner evaluation for fixed-size degree-≤11 polynomial.
///
/// Evaluates the polynomial Σ_{i=0}^{11} coeffs[i] * r^i using Horner's method.
///
/// # Arguments
/// * `coeffs` - Array of 12 coefficients (lowest degree first)
/// * `r` - The point at which to evaluate the polynomial
///
/// # Returns
/// The value of the polynomial at r
pub fn eval_poly12(coeffs: &[Fq; 12], r: &Fq) -> Fq {
    let mut acc = Fq::zero();
    for i in (0..12).rev() {
        acc *= r;
        acc += coeffs[i];
    }
    acc
}

/// Horner evaluation for arbitrary-degree polynomial.
///
/// Evaluates the polynomial Σ coeffs[i] * r^i using Horner's method.
/// Coefficients are assumed to be in order from lowest to highest degree.
///
/// # Arguments
/// * `coeffs` - Slice of coefficients (lowest degree first)
/// * `r` - The point at which to evaluate the polynomial
///
/// # Returns
/// The value of the polynomial at r
pub fn eval_poly_vec(coeffs: &[Fq], r: &Fq) -> Fq {
    let mut acc = Fq::zero();
    for &c in coeffs.iter().rev() {
        acc *= r;
        acc += c;
    }
    acc
}

/// Add polynomial b to polynomial a in place.
///
/// If b has more terms than a, a is extended with zeros as needed.
///
/// # Arguments
/// * `a` - The polynomial to modify (coefficients lowest degree first)
/// * `b` - The polynomial to add (coefficients lowest degree first)
pub fn poly_add_in_place(a: &mut Vec<Fq>, b: &[Fq]) {
    if b.len() > a.len() {
        a.resize(b.len(), Fq::zero());
    }
    for i in 0..b.len() {
        a[i] += b[i];
    }
}

/// Subtract polynomial b from polynomial a in place.
///
/// If b has more terms than a, a is extended with zeros as needed.
///
/// # Arguments
/// * `a` - The polynomial to modify (coefficients lowest degree first)
/// * `b` - The polynomial to subtract (coefficients lowest degree first)
pub fn poly_sub_in_place(a: &mut Vec<Fq>, b: &[Fq]) {
    if b.len() > a.len() {
        a.resize(b.len(), Fq::zero());
    }
    for i in 0..b.len() {
        a[i] -= b[i];
    }
}

/// Multiply two polynomials using convolution.
///
/// Returns a polynomial of degree deg(a) + deg(b).
///
/// # Arguments
/// * `a` - First polynomial (coefficients lowest degree first)
/// * `b` - Second polynomial (coefficients lowest degree first)
///
/// # Returns
/// The product polynomial with length a.len() + b.len() - 1
pub fn poly_mul(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut out = vec![Fq::zero(); a.len() + b.len() - 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            out[i + j] += a[i] * b[j];
        }
    }
    out
}

/// Polynomial long division by a monic divisor.
///
/// Divides the dividend polynomial by a monic divisor g using long division.
/// The divisor g must have leading coefficient 1.
///
/// # Arguments
/// * `dividend` - The polynomial to divide (coefficients lowest degree first)
/// * `g` - The monic divisor polynomial (coefficients lowest degree first)
///
/// # Returns
/// A tuple (quotient, remainder) where deg(remainder) < deg(g)
///
/// # Panics
/// Panics if g is empty or not monic (leading coefficient ≠ 1)
pub fn poly_div_rem_monic(mut dividend: Vec<Fq>, g: &[Fq]) -> (Vec<Fq>, Vec<Fq>) {
    assert!(!g.is_empty(), "divisor g must be non-empty");
    assert!(g.last().unwrap().is_one(), "divisor g must be monic (leading coefficient = 1)");
    
    if dividend.is_empty() || dividend.len() < g.len() {
        return (vec![], dividend);
    }
    
    let n = dividend.len() - 1;
    let m = g.len() - 1; // deg g
    let mut q = vec![Fq::zero(); n - m + 1];
    
    for k in (m..=n).rev() {
        let lead = dividend[k]; // since g is monic, this is the quotient coefficient
        q[k - m] = lead;
        if lead.is_zero() {
            continue;
        }
        // subtract lead * x^{k-m} * g from dividend
        for j in 0..=m {
            dividend[k - m + j] -= lead * g[j];
        }
    }
    
    // trim trailing zeros from remainder
    while let Some(true) = dividend.last().map(|c| c.is_zero()) {
        dividend.pop();
    }
    
    (q, dividend)
}

/// Build the coefficients for g(X) = X^12 - 18 X^6 + 82.
///
/// Returns a vector of coefficients in lowest-degree-first order.
pub fn g_coeffs() -> Vec<Fq> {
    let mut g = vec![Fq::zero(); 13];
    g[0] = Fq::from(82u64);
    g[6] = -Fq::from(18u64);
    g[12] = Fq::one();
    g
}