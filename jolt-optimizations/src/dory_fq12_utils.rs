//! Mock Dory workload for GT ops

use ark_bn254::{Fq, Fq12};
use ark_ff::{BigInteger, Field, One, PrimeField, UniformRand, Zero};
use ark_std::test_rng;

use crate::batched_expressions::{Expression, ExpressionTerm};
use crate::{fq12_to_poly12_coeffs, g_coeffs, poly_div_rem_monic};

/// Helper function to compute a^exp in Fq12
pub fn pow_fq12(base: &Fq12, exp: Fq) -> Fq12 {
    if exp.is_zero() {
        return Fq12::one();
    }
    let exp_bigint = exp.into_bigint();
    base.pow(&exp_bigint)
}

#[derive(Clone)]
pub struct DoryState {
    pub c: Fq12,
    pub d1: Fq12,
    pub d2: Fq12,
    // Protocol parameters
    pub chi: Vec<Fq12>,
    pub c_plus: Fq12,
    pub c_minus: Fq12,
    pub d1l: Fq12,
    pub d1r: Fq12,
    pub d2l: Fq12,
    pub d2r: Fq12,
    pub delta_1l: Fq12,
    pub delta_1r: Fq12,
    pub delta_2l: Fq12,
    pub delta_2r: Fq12,
    pub h_t: Fq12,
    pub e_h1_e2: Fq12,
    pub e_e1_h2: Fq12,
    pub e_h1_gamma2: Fq12,
    pub e_gamma1_h2: Fq12,
}

impl DoryState {
    pub fn random(num_rounds: usize) -> Self {
        let mut rng = test_rng();
        DoryState {
            c: Fq12::rand(&mut rng),
            d1: Fq12::rand(&mut rng),
            d2: Fq12::rand(&mut rng),
            chi: (0..num_rounds).map(|_| Fq12::rand(&mut rng)).collect(),
            c_plus: Fq12::rand(&mut rng),
            c_minus: Fq12::rand(&mut rng),
            d1l: Fq12::rand(&mut rng),
            d1r: Fq12::rand(&mut rng),
            d2l: Fq12::rand(&mut rng),
            d2r: Fq12::rand(&mut rng),
            delta_1l: Fq12::rand(&mut rng),
            delta_1r: Fq12::rand(&mut rng),
            delta_2l: Fq12::rand(&mut rng),
            delta_2r: Fq12::rand(&mut rng),
            h_t: Fq12::rand(&mut rng),
            e_h1_e2: Fq12::rand(&mut rng),
            e_e1_h2: Fq12::rand(&mut rng),
            e_h1_gamma2: Fq12::rand(&mut rng),
            e_gamma1_h2: Fq12::rand(&mut rng),
        }
    }
    
    /// Compute Dory C update in Fq12
    /// C' ← C + χ_i + β·D_2 + β^{-1}·D_1 + α·C_+ + α^{-1}·C_-
    pub fn compute_c_update(&self, round: usize, alpha: Fq, beta: Fq) -> Fq12 {
        let alpha_inv = alpha.inverse().unwrap();
        let beta_inv = beta.inverse().unwrap();
        
        self.c 
            + self.chi[round]
            + pow_fq12(&self.d2, beta)
            + pow_fq12(&self.d1, beta_inv)
            + pow_fq12(&self.c_plus, alpha)
            + pow_fq12(&self.c_minus, alpha_inv)
    }
    
    /// Compute Dory D1 update in Fq12
    /// D_1' ← α·D_{1L} + D_{1R} + αβ·Δ_{1L} + β·Δ_{1R}
    pub fn compute_d1_update(&self, alpha: Fq, beta: Fq) -> Fq12 {
        pow_fq12(&self.d1l, alpha)
            + self.d1r
            + pow_fq12(&self.delta_1l, alpha * beta)
            + pow_fq12(&self.delta_1r, beta)
    }
    
    /// Compute Dory D2 update in Fq12
    /// D_2' ← α^{-1}·D_{2L} + D_{2R} + α^{-1}β^{-1}·Δ_{2L} + β^{-1}·Δ_{2R}
    pub fn compute_d2_update(&self, alpha: Fq, beta: Fq) -> Fq12 {
        let alpha_inv = alpha.inverse().unwrap();
        let beta_inv = beta.inverse().unwrap();
        
        pow_fq12(&self.d2l, alpha_inv)
            + self.d2r
            + pow_fq12(&self.delta_2l, alpha_inv * beta_inv)
            + pow_fq12(&self.delta_2r, beta_inv)
    }
    
    /// Compute Dory C fold in Fq12
    /// C' ← C + s̃_1·s̃_2·H_T + γ·e(H_1, E_2) + γ^{-1}·e(E_1, H_2)
    pub fn compute_c_fold(&self, gamma: Fq, s1_tilde: Fq, s2_tilde: Fq) -> Fq12 {
        let gamma_inv = gamma.inverse().unwrap();
        
        self.c
            + pow_fq12(&self.h_t, s1_tilde * s2_tilde)
            + pow_fq12(&self.e_h1_e2, gamma)
            + pow_fq12(&self.e_e1_h2, gamma_inv)
    }
    
    /// Compute Dory D1 fold in Fq12
    /// D_1' ← D_1 + e(H_1, Γ_{2,0}·s̃_1·γ)
    pub fn compute_d1_fold(&self, gamma: Fq, s1_tilde: Fq) -> Fq12 {
        self.d1 + pow_fq12(&self.e_h1_gamma2, s1_tilde * gamma)
    }
    
    /// Compute Dory D2 fold in Fq12
    /// D_2' ← D_2 + e(Γ_{1,0}·s̃_2·γ^{-1}, H_2)
    pub fn compute_d2_fold(&self, gamma: Fq, s2_tilde: Fq) -> Fq12 {
        let gamma_inv = gamma.inverse().unwrap();
        self.d2 + pow_fq12(&self.e_gamma1_h2, s2_tilde * gamma_inv)
    }
    
    /// Generate all Dory expressions for one round with proper quotients
    pub fn generate_round_expressions(
        &self,
        round: usize,
        alpha: Fq,
        beta: Fq,
        gamma: Fq,
        s1_tilde: Fq,
        s2_tilde: Fq,
    ) -> Vec<Expression> {
        let mut expressions = Vec::new();
        
        let alpha_inv = alpha.inverse().unwrap();
        let beta_inv = beta.inverse().unwrap();
        let gamma_inv = gamma.inverse().unwrap();
        
        // C update expression
        let c_new = self.compute_c_update(round, alpha, beta);
        expressions.push(create_dory_expression_with_quotient(
            format!("C_update_round_{}", round),
            c_new,
            vec![
                (self.c, Fq::one()),
                (self.chi[round], Fq::one()),
                (self.d2, beta),
                (self.d1, beta_inv),
                (self.c_plus, alpha),
                (self.c_minus, alpha_inv),
            ],
        ));
        
        // D1 update expression
        let d1_new = self.compute_d1_update(alpha, beta);
        expressions.push(create_dory_expression_with_quotient(
            format!("D1_update_round_{}", round),
            d1_new,
            vec![
                (self.d1l, alpha),
                (self.d1r, Fq::one()),
                (self.delta_1l, alpha * beta),
                (self.delta_1r, beta),
            ],
        ));
        
        // D2 update expression
        let d2_new = self.compute_d2_update(alpha, beta);
        expressions.push(create_dory_expression_with_quotient(
            format!("D2_update_round_{}", round),
            d2_new,
            vec![
                (self.d2l, alpha_inv),
                (self.d2r, Fq::one()),
                (self.delta_2l, alpha_inv * beta_inv),
                (self.delta_2r, beta_inv),
            ],
        ));
        
        // C fold expression
        let c_fold = self.compute_c_fold(gamma, s1_tilde, s2_tilde);
        expressions.push(create_dory_expression_with_quotient(
            format!("C_fold_round_{}", round),
            c_fold,
            vec![
                (self.c, Fq::one()),
                (self.h_t, s1_tilde * s2_tilde),
                (self.e_h1_e2, gamma),
                (self.e_e1_h2, gamma_inv),
            ],
        ));
        
        // D1 fold expression
        let d1_fold = self.compute_d1_fold(gamma, s1_tilde);
        expressions.push(create_dory_expression_with_quotient(
            format!("D1_fold_round_{}", round),
            d1_fold,
            vec![
                (self.d1, Fq::one()),
                (self.e_h1_gamma2, s1_tilde * gamma),
            ],
        ));
        
        // D2 fold expression
        let d2_fold = self.compute_d2_fold(gamma, s2_tilde);
        expressions.push(create_dory_expression_with_quotient(
            format!("D2_fold_round_{}", round),
            d2_fold,
            vec![
                (self.d2, Fq::one()),
                (self.e_gamma1_h2, s2_tilde * gamma_inv),
            ],
        ));
        
        expressions
    }
}

/// Generate a Dory expression with proper quotient computed from Fq12 values
pub fn create_dory_expression_with_quotient(
    name: String,
    lhs_fq12: Fq12,
    rhs_fq12: Vec<(Fq12, Fq)>,
) -> Expression {
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

    // Compute the product in Fq12 (much more efficient than polynomial exponentiation)
    let mut product = Fq12::one();
    for (elem, exp) in &rhs_fq12 {
        if !exp.is_zero() {
            product *= pow_fq12(elem, *exp);
        }
    }

    // The expression should satisfy: lhs_fq12 = product
    // So the residual is: lhs_fq12 - product = 0 (for valid expressions)
    // But we need the quotient such that: (lhs_poly - product_poly) = quotient * g

    // Get product as polynomial
    let product_poly = fq12_to_poly12_coeffs(&product);

    // Compute polynomial product (without exponentiation, just for the quotient)
    // For this, we multiply the polynomials representing the Fq12 product
    let prod_as_poly = product_poly.to_vec();

    // Since we're in Fq12, lhs_poly and prod_as_poly differ by a multiple of g
    // when viewed as polynomials in Fq[X]

    // For a valid expression, lhs_fq12 = product in Fq12
    // This means lhs_poly ≡ product_poly (mod g)
    // So residual = lhs_poly - product_poly should be divisible by g

    let mut residual = lhs_poly.to_vec();
    for i in 0..prod_as_poly.len().min(residual.len()) {
        residual[i] -= prod_as_poly[i];
    }

    // If prod_as_poly is longer, extend residual
    if prod_as_poly.len() > residual.len() {
        residual.resize(prod_as_poly.len(), Fq::zero());
        for i in lhs_poly.len()..prod_as_poly.len() {
            residual[i] = -prod_as_poly[i];
        }
    }

    // For valid Dory expressions, residual should be 0 or very small
    // But for the general case, compute quotient
    let g = g_coeffs();
    let (quotient, _remainder) = poly_div_rem_monic(residual, &g);

    // For valid expressions, remainder should be zero
    // We'll use the quotient even if remainder is non-zero (for testing tampering)

    Expression {
        name,
        lhs: lhs_poly,
        rhs: rhs_terms,
        quotient: Some(quotient),
    }
}
