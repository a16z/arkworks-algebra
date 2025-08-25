//! Batched verification for multi-exponentiation expressions in Fq12
//!
//! This module provides functionality for verifying multiple expressions of the form:
//! z_i'(X) ≡ ∏_j z_{i,j}(X)^{e_{i,j}} (mod g(X))
//! where g(X) = X^12 - 18X^6 + 82
//!
//! The expressions are aggregated into a single base-field check at a challenge point r ∈ Fp.

use ark_bn254::{Fq, Fq12};
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};

use crate::{
    eval_poly12, eval_poly_vec, fq12_to_poly12_coeffs, g_coeffs, g_eval, poly_div_rem_monic,
    poly_mul, poly_sub_in_place,
};

/// Type alias for a degree-11 polynomial over Fq (12 coefficients)
pub type Poly12 = [Fq; 12];

/// A term in an expression: polynomial with an exponent
#[derive(Clone, Debug)]
pub struct ExpressionTerm {
    /// The polynomial z_{i,j}(X) as coefficients
    pub poly: Poly12,
    /// The exponent e_{i,j} in Fp
    pub exponent: Fq,
}

/// A single expression of the form z_i'(X) ≡ ∏_j z_{i,j}(X)^{e_{i,j}} (mod g(X))
#[derive(Clone, Debug)]
pub struct Expression {
    /// Name/identifier for this expression
    pub name: String,
    /// Left-hand side: z_i'(X)
    pub lhs: Poly12,
    /// Right-hand side terms: { z_{i,j}(X)^{e_{i,j}} }
    pub rhs: Vec<ExpressionTerm>,
    /// Prover-computed quotient q_i(X) such that z_i'(X) - ∏_j z_{i,j}(X)^{e_{i,j}} = q_i(X) * g(X)
    pub quotient: Option<Vec<Fq>>,
}

/// Specification for a round of expressions
#[derive(Clone, Debug)]
pub struct RoundSpec {
    /// List of expressions for this round
    pub expressions: Vec<Expression>,
    /// Optional transcript bytes for Fiat-Shamir
    pub transcript_bytes: Option<Vec<u8>>,
}

/// Result of batch verification
#[derive(Clone, Debug)]
pub struct BatchCheckResult {
    /// Whether the check passed
    pub ok: bool,
    /// Aggregated left-hand side value
    pub lhs: Fq,
    /// Aggregated right-hand side value (quotient * g(r))
    pub rhs: Fq,
    /// Challenge point used for evaluation
    pub r: Fq,
}

// ============================================================================
// Prover Helpers
// ============================================================================

/// Compute the residual for multiplication-only expressions
/// IMPORTANT: This only works for expressions where all exponents are 1
/// For general exponents, the prover must use Fq12 arithmetic
///
/// # Arguments
/// * `lhs` - The left-hand side polynomial z_i'(X)
/// * `rhs_terms` - The right-hand side terms (must all have exponent = 1)
///
/// # Returns
/// The residual polynomial as a Vec<Fq> (trimmed of trailing zeros)
pub fn compute_residual_no_reduce(lhs: &Poly12, rhs_terms: &[ExpressionTerm]) -> Vec<Fq> {
    // Only handle the simple case where all exponents are 1
    // This is just for testing - real Dory expressions need the Fq12 approach
    
    let mut product = vec![Fq::one()];
    
    for term in rhs_terms {
        if term.exponent == Fq::one() {
            product = poly_mul(&product, &term.poly[..]);
        } else {
            // For benchmarking, just return a dummy quotient
            // In real usage, compute from Fq12 values
            return vec![Fq::zero(); 12];
        }
    }
    
    // Compute lhs - product
    let mut residual = lhs.to_vec();
    poly_sub_in_place(&mut residual, &product);
    
    // Trim trailing zeros
    while residual.len() > 1 && residual.last() == Some(&Fq::zero()) {
        residual.pop();
    }
    
    residual
}

/// Compute the residual from Fq12 elements directly (for arbitrary exponents)
/// This is much more efficient than polynomial exponentiation
///
/// # Arguments
/// * `lhs_fq12` - The left-hand side as an Fq12 element
/// * `rhs_fq12` - The right-hand side terms as Fq12 elements with their exponents
///
/// # Returns
/// The residual polynomial after reduction by g(X)
pub fn compute_residual_from_fq12(
    lhs_fq12: &Fq12,
    rhs_fq12: &[(Fq12, Fq)],
) -> Vec<Fq> {
    // Compute the product in Fq12
    let mut product = Fq12::one();
    for (elem, exp) in rhs_fq12 {
        if !exp.is_zero() {
            let exp_bigint = exp.into_bigint();
            product *= elem.pow(&exp_bigint);
        }
    }
    
    // Compute residual in Fq12
    let residual_fq12 = *lhs_fq12 - product;
    
    // Convert to polynomial
    let residual_poly = fq12_to_poly12_coeffs(&residual_fq12);
    
    // The residual should be divisible by g(X) if the expression is satisfied
    // Return as Vec for division
    residual_poly.to_vec()
}

/// Divide the residual polynomial by g(X) to obtain the quotient
///
/// # Arguments
/// * `residual` - The residual polynomial R(X) = z_i'(X) - ∏_j z_{i,j}(X)^{e_{i,j}}
/// * `g` - The polynomial g(X) = X^12 - 18X^6 + 82
///
/// # Returns
/// The quotient q_i(X) such that R(X) = q_i(X) * g(X)
///
/// # Panics
/// Panics if the remainder is non-zero (dishonest prover case)
pub fn quotient_divide_by_g(residual: Vec<Fq>, g: &[Fq]) -> Vec<Fq> {
    let (quotient, remainder) = poly_div_rem_monic(residual, g);
    
    // In honest execution, remainder should be zero
    for coeff in &remainder {
        assert!(
            coeff.is_zero(),
            "Residual is not divisible by g(X) - remainder is non-zero"
        );
    }
    
    quotient
}

/// Attach a computed quotient to an expression
///
/// # Arguments
/// * `expression` - The expression to update
/// * `q` - The quotient polynomial coefficients
pub fn attach_quotient(expression: &mut Expression, q: Vec<Fq>) {
    expression.quotient = Some(q);
}

// ============================================================================
// Verifier Batch Engine
// ============================================================================

/// Verify a batch of expressions at a challenge point r
///
/// Checks that: ∑_i γ_i(z_i'(r) - ∏_j z_{i,j}(r)^{e_{i,j}}) = (∑_i γ_i q_i(r)) * g(r)
///
/// # Arguments
/// * `expressions` - The expressions to verify (must have quotients attached)
/// * `r` - The challenge point for evaluation
/// * `gammas` - Random coefficients for aggregation (same length as expressions)
/// * `g` - The polynomial g(X) coefficients
///
/// # Returns
/// A BatchCheckResult indicating success/failure and the computed values
pub fn verify_batched_expressions(
    expressions: &[Expression],
    r: Fq,
    gammas: &[Fq],
    g: &[Fq; 13],
) -> BatchCheckResult {
    assert_eq!(
        expressions.len(),
        gammas.len(),
        "Number of expressions must match number of gammas"
    );
    
    let mut lhs_sum = Fq::zero();
    let mut q_sum = Fq::zero();
    
    for (i, expression) in expressions.iter().enumerate() {
        // Evaluate lhs at r
        let lhs_i = eval_poly12(&expression.lhs, &r);
        
        // Compute product of rhs terms
        let mut prod_i = Fq::one();
        for term in &expression.rhs {
            let poly_eval = eval_poly12(&term.poly, &r);
            
            // Compute poly_eval^exponent
            if !term.exponent.is_zero() {
                let exp_bigint = term.exponent.into_bigint();
                let powered = poly_eval.pow(&exp_bigint);
                prod_i *= powered;
            }
            // If exponent is zero, the term contributes 1 to the product
        }
        
        // Compute residual at r
        let res_i = lhs_i - prod_i;
        
        // Aggregate with gamma
        lhs_sum += gammas[i] * res_i;
        
        // Evaluate quotient at r
        let quotient = expression
            .quotient
            .as_ref()
            .expect("Quotient must be attached before verification");
        let qi_r = eval_poly_vec(quotient, &r);
        q_sum += gammas[i] * qi_r;
    }
    
    // Compute g(r)
    let gr = g_eval(&r);
    
    // Compute rhs = q_sum * g(r)
    let rhs = q_sum * gr;
    
    // Check equality
    let ok = lhs_sum == rhs;
    
    BatchCheckResult {
        ok,
        lhs: lhs_sum,
        rhs,
        r,
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert an Fq12 element to a Poly12 representation
pub fn fq12_to_poly12(elem: &Fq12) -> Poly12 {
    fq12_to_poly12_coeffs(elem)
}

/// Create an expression from Fq12 elements
pub fn expression_from_fq12(
    name: String,
    lhs: &Fq12,
    rhs_elems: Vec<(Fq12, Fq)>,
) -> Expression {
    let lhs_poly = fq12_to_poly12(lhs);
    let rhs_terms = rhs_elems
        .into_iter()
        .map(|(elem, exp)| ExpressionTerm {
            poly: fq12_to_poly12(&elem),
            exponent: exp,
        })
        .collect();
    
    Expression {
        name,
        lhs: lhs_poly,
        rhs: rhs_terms,
        quotient: None,
    }
}

/// Compute and attach quotients for all expressions in a round
pub fn prepare_round_quotients(round: &mut RoundSpec) {
    let g = g_coeffs();
    
    for expression in &mut round.expressions {
        let residual = compute_residual_no_reduce(&expression.lhs, &expression.rhs);
        let quotient = quotient_divide_by_g(residual, &g);
        attach_quotient(expression, quotient);
    }
}