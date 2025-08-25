//! Fq12 to polynomial conversion utilities for BN254
//!
//! This module provides functionality for converting Fq12 field elements
//! to polynomial representations over Fq with the relation g(X) = X^12 - 18 X^6 + 82.

use ark_bn254::{Fq, Fq12};
use ark_ff::{Field, Zero};

/// Flatten ark_bn254::Fq12 to 12 base-field coefficients for a(X)=Σ c_i X^i, X=w,
/// with the relation g(X) = X^12 - 18 X^6 + 82.
///
/// The BN254 Fq12 field is constructed as a tower extension:
/// - Fq2 = Fq[u]/(u^2 + 1)
/// - Fq6 = Fq2[v]/(v^3 - (9 + u))
/// - Fq12 = Fq6[w]/(w^2 - v)
///
/// This function maps an Fq12 element to its polynomial representation
/// in Fq[X] where X = w, using the mapping rule:
/// (x + y·u)·w^k = (x - 9y)·w^k + y·w^{k+6}, for k∈{0..5}.
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