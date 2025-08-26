//! Fq12 polynomial operations and conversions for BN254
//!
//! This module provides:
//! - Conversion between Fq12 field elements and polynomial representations
//! - Polynomial arithmetic operations over Fq[X]
//! - Evaluation and manipulation of the minimal polynomial g(X) = X^12 - 18X^6 + 82

use ark_bn254::{Fq, Fq12};
use ark_ff::{One, Zero};

/// Flatten Fq12 to 12 base-field coefficients for a(X)=Σ c_i X^i, X=w,
/// with the relation g(X) = X^12 - 18 X^6 + 82.
///
/// The BN254 Fq12 field is constructed as a tower extension:
/// - Fq2 = Fq[u]/(u^2 + 1)
/// - Fq6 = Fq2[v]/(v^3 - (9 + u))
/// - Fq12 = Fq6[w]/(w^2 - v)
///
/// This function maps an Fq12 element to its polynomial representation
/// in Fq[X] where X = w, using the mapping:
/// (x + y·u)·w^k = (x - 9y)·w^k + y·w^{k+6}, for k∈{0..5}.
/// @TODO(markosg04) provide proof?
///
/// # Arguments
/// * `a` - An Fq12 field element to convert
///
/// # Returns
/// An array of 12 Fq coefficients representing the polynomial a(X) = Σ c_i X^i
pub fn fq12_to_poly12_coeffs(a: &Fq12) -> [Fq; 12] {
    let nine = Fq::from(9u64);
    let mut c = [Fq::zero(); 12];

    // (term, k) pairs mapping Fq12 basis elements to powers of w:
    // 1, v, v^2, w, v·w, v^2·w  ↔  w^0, w^2, w^4, w^1, w^3, w^5
    let terms = [
        (&a.c0.c0, 0usize), // 1 → w^0
        (&a.c0.c1, 2usize), // v → w^2
        (&a.c0.c2, 4usize), // v^2 → w^4
        (&a.c1.c0, 1usize), // w → w^1
        (&a.c1.c1, 3usize), // v·w → w^3
        (&a.c1.c2, 5usize), // v^2·w → w^5
    ];

    for (fp2, k) in terms {
        let x = fp2.c0; // coefficient of 1 in Fp2
        let y = fp2.c1; // coefficient of u in Fp2 (with u^2 = -1)
                        // Apply the mapping: (x + y·u)·w^k = (x - 9y)·w^k + y·w^{k+6}
        c[k] += x - nine * y;
        c[k + 6] += y;
    }
    c
}

// ============================================================================
// Polynomial Operations
// ============================================================================

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
    let r2 = r.square(); // r^2
    let r3 = r2 * r; // r^3
    let r6 = r3.square(); // r^6
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
    assert!(
        g.last().unwrap().is_one(),
        "divisor g must be monic (leading coefficient = 1)"
    );

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
