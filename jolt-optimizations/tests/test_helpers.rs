use ark_bn254::{Fq, Fq12};
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};

use jolt_optimizations::{
    batched_expressions::{Expression, ExpressionTerm},
    fq12_to_poly12_coeffs, g_coeffs, poly_div_rem_monic, poly_mul,
};

/// Helper to compute polynomial exponentiation without modular reduction
/// This is VERY expensive but needed for correct quotient computation
fn poly_pow_no_mod(poly: &[Fq], exp: Fq) -> Vec<Fq> {
    if exp.is_zero() {
        return vec![Fq::one()];
    }
    
    let exp_bigint = exp.into_bigint();
    let exp_limbs = exp_bigint.as_ref();
    
    let mut result = vec![Fq::one()];
    let mut base = poly.to_vec();
    
    for limb in exp_limbs {
        let mut limb_val = *limb;
        for _ in 0..64 {
            if limb_val & 1 == 1 {
                result = poly_mul(&result, &base);
            }
            if limb_val > 1 {
                let base_squared = poly_mul(&base, &base);
                base = base_squared;
            }
            limb_val >>= 1;
            if limb_val == 0 {
                break;
            }
        }
    }
    
    result
}

/// Helper to compute polynomial exponentiation mod g(X)
/// This is expensive but correct for testing
fn poly_pow_mod_g(poly: &[Fq], exp: Fq, g: &[Fq]) -> Vec<Fq> {
    if exp.is_zero() {
        return vec![Fq::one()];
    }
    
    let exp_bigint = exp.into_bigint();
    let exp_limbs = exp_bigint.as_ref();
    
    let mut result = vec![Fq::one()];
    let mut base = poly.to_vec();
    
    for limb in exp_limbs {
        let mut limb_val = *limb;
        for _ in 0..64 {
            if limb_val & 1 == 1 {
                result = poly_mul(&result, &base);
                // Reduce mod g
                let (_, rem) = poly_div_rem_monic(result, g);
                result = rem;
            }
            if limb_val > 1 {
                base = poly_mul(&base, &base);
                // Reduce mod g
                let (_, rem) = poly_div_rem_monic(base.clone(), g);
                base = rem;
            }
            limb_val >>= 1;
        }
    }
    
    result
}

/// Create an expression with proper polynomial quotient (for testing)
/// This computes the actual polynomial operations, which is expensive but correct
pub fn create_test_expression_with_quotient(
    name: String,
    lhs_fq12: Fq12,
    rhs_fq12: Vec<(Fq12, Fq)>,
) -> Expression {
    let g = g_coeffs();
    
    // Convert to polynomial form
    let lhs_poly = fq12_to_poly12_coeffs(&lhs_fq12);
    
    // Create expression terms
    let rhs_terms: Vec<ExpressionTerm> = rhs_fq12
        .iter()
        .map(|(elem, exp)| ExpressionTerm {
            poly: fq12_to_poly12_coeffs(elem),
            exponent: *exp,
        })
        .collect();
    
    // Compute the product as polynomials WITHOUT reducing mod g
    // We need the full polynomial to compute the correct quotient
    let mut product = vec![Fq::one()];
    
    for term in rhs_terms.iter() {
        // Compute term^exp without mod g reduction
        let term_pow = poly_pow_no_mod(&term.poly.to_vec(), term.exponent);
        product = poly_mul(&product, &term_pow);
    }
    
    // Compute residual
    let mut residual = lhs_poly.to_vec();
    
    // Extend to match lengths
    let max_len = residual.len().max(product.len());
    residual.resize(max_len, Fq::zero());
    let mut product_extended = product;
    product_extended.resize(max_len, Fq::zero());
    
    for i in 0..max_len {
        residual[i] -= product_extended[i];
    }
    
    // Trim trailing zeros
    while residual.len() > 1 && residual.last() == Some(&Fq::zero()) {
        residual.pop();
    }
    
    // Divide by g to get quotient
    let (quotient, remainder) = poly_div_rem_monic(residual, &g);
    
    // For debugging: check if remainder is non-zero (shouldn't happen for valid expressions)
    debug_assert!(
        remainder.iter().all(|x| x.is_zero()),
        "Non-zero remainder in quotient division"
    );
    
    Expression {
        name,
        lhs: lhs_poly,
        rhs: rhs_terms,
        quotient: Some(quotient),
    }
}