//! Direct polynomial quotient computation for Schwartz-Zippel verification
//! 
//! This module provides functionality to compute quotient polynomials directly in F[x]
//! rather than computing in Fp12 first. This is necessary for correctly handling
//! arbitrary exponents in the batched expression verification scheme.

use ark_bn254::Fq;
use ark_ff::{One, PrimeField, Zero};

/// Multiply two polynomials and reduce modulo g(X)
/// g(X) = X^12 - 18X^6 + 82
fn poly_mul_mod_g(a: &[Fq], b: &[Fq], g: &[Fq]) -> Vec<Fq> {
    if a.is_empty() || b.is_empty() {
        return vec![Fq::zero()];
    }
    
    // Standard polynomial multiplication
    let mut product = vec![Fq::zero(); a.len() + b.len() - 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            product[i + j] += a[i] * b[j];
        }
    }
    
    // Reduce modulo g
    let (_, remainder) = poly_div_rem_monic(product, g);
    remainder
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

/// Compute polynomial exponentiation modulo g(X) using repeated squaring
/// Computes a(X)^e mod g(X)
fn poly_pow_mod_g(base: &[Fq], exponent: &Fq, g: &[Fq]) -> Vec<Fq> {
    if exponent.is_zero() {
        return vec![Fq::one()]; // a^0 = 1
    }
    
    let exp_bits = exponent.into_bigint();
    let mut result = vec![Fq::one()]; // Start with 1
    let mut base_power = base.to_vec();
    
    // Repeated squaring
    for limb in exp_bits.as_ref() {
        let mut bit_mask = 1u64;
        for _ in 0..64 {
            if limb & bit_mask != 0 {
                result = poly_mul_mod_g(&result, &base_power, g);
            }
            base_power = poly_mul_mod_g(&base_power, &base_power, g);
            bit_mask <<= 1;
        }
    }
    
    result
}

/// Compute the product of multiple polynomial powers modulo g(X)
/// Computes ∏_j poly_j(X)^{exp_j} mod g(X)
fn compute_product_mod_g(terms: &[(Vec<Fq>, Fq)], g: &[Fq]) -> Vec<Fq> {
    if terms.is_empty() {
        return vec![Fq::one()];
    }
    
    let mut product = vec![Fq::one()];
    
    for (poly, exp) in terms {
        if !exp.is_zero() {
            let powered = poly_pow_mod_g(poly, exp, g);
            product = poly_mul_mod_g(&product, &powered, g);
        }
        // If exp is zero, term contributes 1 (identity), so we skip
    }
    
    product
}

/// Subtract two polynomials
fn poly_sub(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    let len = a.len().max(b.len());
    let mut result = vec![Fq::zero(); len];
    
    for i in 0..a.len() {
        result[i] += a[i];
    }
    for i in 0..b.len() {
        result[i] -= b[i];
    }
    
    // Remove leading zeros
    while result.len() > 1 && result.last().unwrap().is_zero() {
        result.pop();
    }
    
    result
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
    // Compute RHS product mod g
    let rhs_product = compute_product_mod_g(rhs_terms, g);
    
    // Compute residual: LHS - RHS
    let residual = poly_sub(lhs, &rhs_product);
    
    // Divide residual by g to get quotient
    let (quotient, remainder) = poly_div_rem_monic(residual, g);
    
    // Verify remainder is zero (should be in honest execution)
    for coeff in &remainder {
        if !coeff.is_zero() {
            return Err(format!(
                "Residual is not divisible by g(X) - remainder is non-zero: {:?}",
                remainder
            ));
        }
    }
    
    Ok(quotient)
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
    fn test_poly_mul_mod_g() {
        let g = g_coeffs();
        
        // Test: (X)(X) mod g = X^2
        let a = vec![Fq::zero(), Fq::one()]; // X
        let b = vec![Fq::zero(), Fq::one()]; // X
        let result = poly_mul_mod_g(&a, &b, &g);
        
        // X * X = X^2, which is degree < 12, so no reduction needed
        assert_eq!(result.len(), 3);
        assert!(result[0].is_zero());
        assert!(result[1].is_zero());
        assert!(result[2].is_one());
    }
    
    #[test]
    fn test_poly_pow_mod_g() {
        let g = g_coeffs();
        
        // Test: X^2 mod g
        let x = vec![Fq::zero(), Fq::one()]; // X
        let exp = Fq::from(2u64);
        let result = poly_pow_mod_g(&x, &exp, &g);
        
        // X^2 should be [0, 0, 1]
        assert_eq!(result.len(), 3);
        assert!(result[0].is_zero());
        assert!(result[1].is_zero());
        assert!(result[2].is_one());
        
        // Test: X^0 = 1
        let exp_zero = Fq::zero();
        let result_zero = poly_pow_mod_g(&x, &exp_zero, &g);
        assert_eq!(result_zero, vec![Fq::one()]);
    }
    
    #[test]
    fn test_compute_quotient_simple() {
        let mut rng = test_rng();
        let g = g_coeffs();
        
        // Create a simple polynomial that's divisible by g
        // Let's say q(X) = X^2 + 3X + 5
        let q = vec![Fq::from(5u64), Fq::from(3u64), Fq::one()];
        
        // Compute lhs = q(X) * g(X)
        let mut lhs = vec![Fq::zero(); q.len() + g.len() - 1];
        for i in 0..q.len() {
            for j in 0..g.len() {
                lhs[i + j] += q[i] * g[j];
            }
        }
        
        // RHS is empty (equals 1), so residual = lhs - 1
        let rhs_terms: Vec<(Vec<Fq>, Fq)> = vec![];
        
        // But we need to adjust - let's make lhs = q*g + 1 so residual = q*g
        lhs[0] += Fq::one();
        
        // Now compute quotient - should get back q
        let computed_q = compute_quotient_direct(&lhs, &rhs_terms, &g);
        
        // Actually, let's test a correct case
        // Set lhs = q * g exactly
        let mut lhs_correct = vec![Fq::zero(); q.len() + g.len() - 1];
        for i in 0..q.len() {
            for j in 0..g.len() {
                lhs_correct[i + j] += q[i] * g[j];
            }
        }
        
        // RHS = 0, so residual = lhs = q * g
        let empty_rhs: Vec<(Vec<Fq>, Fq)> = vec![];
        let one_poly = vec![Fq::one()];
        let rhs_with_one = vec![(one_poly, Fq::one())]; // RHS = 1
        
        // LHS = q*g + 1, RHS = 1, so residual = q*g
        lhs_correct[0] += Fq::one();
        let computed_q2 = compute_quotient_direct(&lhs_correct, &rhs_with_one, &g).unwrap();
        
        // Should get back q
        assert_eq!(computed_q2.len(), q.len());
        for i in 0..q.len() {
            assert_eq!(computed_q2[i], q[i], "Quotient mismatch at index {}", i);
        }
    }
    
    #[test]
    fn test_compute_quotient_with_powers() {
        let g = g_coeffs();
        
        // Test: (X^2 + 1)^2 = X^4 + 2X^2 + 1
        let poly = vec![Fq::one(), Fq::zero(), Fq::one()]; // X^2 + 1
        let exp = Fq::from(2u64);
        
        // Compute (X^2 + 1)^2 directly
        let squared = poly_pow_mod_g(&poly, &exp, &g);
        
        // Set this as LHS
        let lhs = squared.clone();
        
        // RHS is the original polynomial with exponent 2
        let rhs_terms = vec![(poly.clone(), exp)];
        
        // Quotient should be zero since LHS = RHS mod g
        let quotient = compute_quotient_direct(&lhs, &rhs_terms, &g).unwrap();
        
        // Check that quotient is zero (or very small due to the mod g reduction)
        for coeff in &quotient {
            assert!(coeff.is_zero(), "Quotient should be zero for matching expressions");
        }
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
        
        // Large exponent
        let exp = Fq::from(17u64);
        
        // Compute poly^17 mod g
        let powered = poly_pow_mod_g(&poly, &exp, &g);
        
        // Verify it's reduced (degree < 12)
        assert!(powered.len() <= 12, "Result should be reduced mod g");
        
        // Set as LHS and verify quotient computation
        let lhs = powered.clone();
        let rhs_terms = vec![(poly.clone(), exp)];
        
        let quotient = compute_quotient_direct(&lhs, &rhs_terms, &g).unwrap();
        
        // Should be zero since LHS = RHS mod g
        for coeff in &quotient {
            assert!(coeff.is_zero(), "Quotient should be zero for matching expressions");
        }
    }
    
    #[test]
    fn test_multiple_terms_product() {
        let mut rng = test_rng();
        let g = g_coeffs();
        
        // Create three random polynomials
        let a = vec![Fq::rand(&mut rng), Fq::rand(&mut rng), Fq::rand(&mut rng)];
        let b = vec![Fq::rand(&mut rng), Fq::rand(&mut rng)];
        let c = vec![Fq::rand(&mut rng), Fq::rand(&mut rng), Fq::rand(&mut rng), Fq::rand(&mut rng)];
        
        // Compute a^2 * b^3 * c mod g
        let a_squared = poly_pow_mod_g(&a, &Fq::from(2u64), &g);
        let b_cubed = poly_pow_mod_g(&b, &Fq::from(3u64), &g);
        let temp = poly_mul_mod_g(&a_squared, &b_cubed, &g);
        let product = poly_mul_mod_g(&temp, &c, &g);
        
        // Set as LHS
        let lhs = product.clone();
        
        // RHS terms
        let rhs_terms = vec![
            (a.clone(), Fq::from(2u64)),
            (b.clone(), Fq::from(3u64)),
            (c.clone(), Fq::one()),
        ];
        
        // Compute quotient - should be zero
        let quotient = compute_quotient_direct(&lhs, &rhs_terms, &g).unwrap();
        
        for coeff in &quotient {
            assert!(coeff.is_zero(), "Quotient should be zero for matching expressions");
        }
    }
    
    #[test]
    fn test_non_zero_quotient() {
        let g = g_coeffs();
        
        // Create LHS that's NOT equal to RHS
        // LHS = X^3 + X + 1
        let lhs = vec![Fq::one(), Fq::one(), Fq::zero(), Fq::one()];
        
        // RHS = X^2
        let x_squared = vec![Fq::zero(), Fq::zero(), Fq::one()];
        let rhs_terms = vec![(x_squared, Fq::one())];
        
        // Residual = X^3 + X + 1 - X^2 = X^3 - X^2 + X + 1
        // This likely won't be divisible by g, so should get an error
        let result = compute_quotient_direct(&lhs, &rhs_terms, &g);
        
        // This should fail since arbitrary polynomials aren't divisible by g
        assert!(result.is_err(), "Arbitrary difference shouldn't be divisible by g");
    }
}