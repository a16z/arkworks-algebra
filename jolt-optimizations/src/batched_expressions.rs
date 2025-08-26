//! This module provides functionality for verifying multiple expressions of the form:
//! z_i'(X) ≡ ∏_j z_{i,j}(X)^{e_{i,j}} (mod g(X))
//! where g(X) = X^12 - 18X^6 + 82

use ark_bn254::Fq;
use ark_ff::{Field, One, PrimeField, Zero};

use crate::{
    compute_quotient::compute_quotient_direct, eval_poly12, eval_poly_vec, g_coeffs, g_eval,
};

pub type Poly12 = [Fq; 12];

#[derive(Clone, Debug)]
pub struct ExpressionTerm {
    pub poly: Poly12,
    pub exponent: Fq,
}

/// A single expression of the form z_i'(X) ≡ ∏_j z_{i,j}(X)^{e_{i,j}} (mod g(X))
#[derive(Clone, Debug)]
pub struct Expression {
    pub name: String,
    /// Left-hand side: z_i'(X)
    pub lhs: Poly12,
    /// Right-hand side terms: { z_{i,j}(X)^{e_{i,j}} }
    pub rhs: Vec<ExpressionTerm>,
    /// Prover-computed quotient q_i(X) such that z_i'(X) - ∏_j z_{i,j}(X)^{e_{i,j}} = q_i(X) * g(X)
    pub quotient: Option<Vec<Fq>>,
}

impl Expression {
    /// Create an expression and compute its quotient directly in F[x]
    pub fn new(name: String, lhs: Poly12, rhs: Vec<ExpressionTerm>) -> Self {
        // Convert to format needed by compute_quotient_direct
        let lhs_vec = lhs.to_vec();
        let rhs_terms: Vec<(Vec<Fq>, Fq)> = rhs
            .iter()
            .map(|term| (term.poly.to_vec(), term.exponent))
            .collect();

        let g = g_coeffs();
        let quotient =
            compute_quotient_direct(&lhs_vec, &rhs_terms, &g).expect("Failed to compute quotient");

        Expression {
            name,
            lhs,
            rhs,
            quotient: Some(quotient),
        }
    }

    /// Create an expression without computing quotient (for testing)
    pub fn without_quotient(name: String, lhs: Poly12, rhs: Vec<ExpressionTerm>) -> Self {
        Expression {
            name,
            lhs,
            rhs,
            quotient: None,
        }
    }

    /// Compute and attach quotient to an expression
    pub fn compute_quotient(&mut self) {
        let lhs_vec = self.lhs.to_vec();
        let rhs_terms: Vec<(Vec<Fq>, Fq)> = self
            .rhs
            .iter()
            .map(|term| (term.poly.to_vec(), term.exponent))
            .collect();

        let g = g_coeffs();
        let quotient =
            compute_quotient_direct(&lhs_vec, &rhs_terms, &g).expect("Failed to compute quotient");

        self.quotient = Some(quotient);
    }
}

#[derive(Clone, Debug)]
pub struct RoundExpresions {
    pub expressions: Vec<Expression>,
}

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

/// Verify a batch of expressions at a challenge point r
///
/// Checks that: ∑_i γ_i(z_i'(r) - ∏_j z_{i,j}(r)^{e_{i,j}}) = (∑_i γ_i q_i(r)) * g(r)
///
/// # Arguments
/// * `expressions` - The expressions to verify (must have quotients attached)
/// * `r` - The challenge point for evaluation
/// * `gammas` - Random coefficients for aggregation (same length as expressions)
///
/// # Returns
/// A BatchCheckResult indicating success/failure and the computed values
pub fn verify_batched_expressions(
    expressions: &[Expression],
    r: Fq,
    gammas: &[Fq],
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

    let gr = g_eval(&r);
    let rhs = q_sum * gr;
    let ok = lhs_sum == rhs;

    BatchCheckResult {
        ok,
        lhs: lhs_sum,
        rhs,
        r,
    }
}
