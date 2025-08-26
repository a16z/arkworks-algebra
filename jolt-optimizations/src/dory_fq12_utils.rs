//! Mock Dory workload for GT ops

use ark_bn254::{Fq, Fq12};
use ark_ff::{Field, One, PrimeField, UniformRand, Zero};
use ark_std::test_rng;

use crate::batched_expressions::{Expression, ExpressionTerm};
use crate::fq12_to_poly12_coeffs;

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

    /// D_1' ← α·D_{1L} + D_{1R} + αβ·Δ_{1L} + β·Δ_{1R}
    pub fn compute_d1_update(&self, alpha: Fq, beta: Fq) -> Fq12 {
        pow_fq12(&self.d1l, alpha)
            + self.d1r
            + pow_fq12(&self.delta_1l, alpha * beta)
            + pow_fq12(&self.delta_1r, beta)
    }

    /// D_2' ← α^{-1}·D_{2L} + D_{2R} + α^{-1}β^{-1}·Δ_{2L} + β^{-1}·Δ_{2R}
    pub fn compute_d2_update(&self, alpha: Fq, beta: Fq) -> Fq12 {
        let alpha_inv = alpha.inverse().unwrap();
        let beta_inv = beta.inverse().unwrap();

        pow_fq12(&self.d2l, alpha_inv)
            + self.d2r
            + pow_fq12(&self.delta_2l, alpha_inv * beta_inv)
            + pow_fq12(&self.delta_2r, beta_inv)
    }

    /// C' ← C + s̃_1·s̃_2·H_T + γ·e(H_1, E_2) + γ^{-1}·e(E_1, H_2)
    pub fn compute_c_fold(&self, gamma: Fq, s1_tilde: Fq, s2_tilde: Fq) -> Fq12 {
        let gamma_inv = gamma.inverse().unwrap();

        self.c
            + pow_fq12(&self.h_t, s1_tilde * s2_tilde)
            + pow_fq12(&self.e_h1_e2, gamma)
            + pow_fq12(&self.e_e1_h2, gamma_inv)
    }

    /// D_1' ← D_1 + e(H_1, Γ_{2,0}·s̃_1·γ)
    pub fn compute_d1_fold(&self, gamma: Fq, s1_tilde: Fq) -> Fq12 {
        self.d1 + pow_fq12(&self.e_h1_gamma2, s1_tilde * gamma)
    }

    /// D_2' ← D_2 + e(Γ_{1,0}·s̃_2·γ^{-1}, H_2)
    pub fn compute_d2_fold(&self, gamma: Fq, s2_tilde: Fq) -> Fq12 {
        let gamma_inv = gamma.inverse().unwrap();
        self.d2 + pow_fq12(&self.e_gamma1_h2, s2_tilde * gamma_inv)
    }

    /// Compute all rounds naively in Fq12 (for benchmarking comparison)
    pub fn compute_all_rounds(
        &self,
        num_rounds: usize,
        alphas: &[Fq],
        betas: &[Fq],
        gammas: &[Fq],
        s1_tildes: &[Fq],
        s2_tildes: &[Fq],
    ) -> (Fq12, Fq12, Fq12) {
        let mut c = self.c;
        let mut d1 = self.d1;
        let mut d2 = self.d2;

        for round in 0..num_rounds {
            let alpha = alphas[round];
            let beta = betas[round];
            let gamma = gammas[round];
            let s1_tilde = s1_tildes[round];
            let s2_tilde = s2_tildes[round];

            c = self.compute_c_update(round, alpha, beta);
            d1 = self.compute_d1_update(alpha, beta);
            d2 = self.compute_d2_update(alpha, beta);

            c = self.compute_c_fold(gamma, s1_tilde, s2_tilde);
            d1 = self.compute_d1_fold(gamma, s1_tilde);
            d2 = self.compute_d2_fold(gamma, s2_tilde);
        }

        (c, d1, d2)
    }

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

        let c_new = self.compute_c_update(round, alpha, beta);
        expressions.push(Expression::new(
            format!("C_update_round_{}", round),
            fq12_to_poly12_coeffs(&c_new),
            vec![
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.c), exponent: Fq::one() },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.chi[round]), exponent: Fq::one() },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.d2), exponent: beta },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.d1), exponent: beta_inv },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.c_plus), exponent: alpha },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.c_minus), exponent: alpha_inv },
            ],
        ));

        let d1_new = self.compute_d1_update(alpha, beta);
        expressions.push(Expression::new(
            format!("D1_update_round_{}", round),
            fq12_to_poly12_coeffs(&d1_new),
            vec![
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.d1l), exponent: alpha },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.d1r), exponent: Fq::one() },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.delta_1l), exponent: alpha * beta },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.delta_1r), exponent: beta },
            ],
        ));

        let d2_new = self.compute_d2_update(alpha, beta);
        expressions.push(Expression::new(
            format!("D2_update_round_{}", round),
            fq12_to_poly12_coeffs(&d2_new),
            vec![
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.d2l), exponent: alpha_inv },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.d2r), exponent: Fq::one() },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.delta_2l), exponent: alpha_inv * beta_inv },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.delta_2r), exponent: beta_inv },
            ],
        ));

        let c_fold = self.compute_c_fold(gamma, s1_tilde, s2_tilde);
        expressions.push(Expression::new(
            format!("C_fold_round_{}", round),
            fq12_to_poly12_coeffs(&c_fold),
            vec![
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.c), exponent: Fq::one() },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.h_t), exponent: s1_tilde * s2_tilde },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.e_h1_e2), exponent: gamma },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.e_e1_h2), exponent: gamma_inv },
            ],
        ));

        let d1_fold = self.compute_d1_fold(gamma, s1_tilde);
        expressions.push(Expression::new(
            format!("D1_fold_round_{}", round),
            fq12_to_poly12_coeffs(&d1_fold),
            vec![
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.d1), exponent: Fq::one() },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.e_h1_gamma2), exponent: s1_tilde * gamma },
            ],
        ));

        let d2_fold = self.compute_d2_fold(gamma, s2_tilde);
        expressions.push(Expression::new(
            format!("D2_fold_round_{}", round),
            fq12_to_poly12_coeffs(&d2_fold),
            vec![
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.d2), exponent: Fq::one() },
                ExpressionTerm { poly: fq12_to_poly12_coeffs(&self.e_gamma1_h2), exponent: s2_tilde * gamma_inv },
            ],
        ));

        expressions
    }
}
