use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use ark_bn254::{Fq, Fq12};
use ark_ff::{BigInteger, Field, One, PrimeField, UniformRand, Zero};
use ark_std::test_rng;

use jolt_optimizations::batched_expressions::{
    verify_batched_expressions, Expression, ExpressionTerm,
};
use jolt_optimizations::dory_fq12_utils::DoryState;
use jolt_optimizations::{fq12_to_poly12_coeffs, g_coeffs};

// Use pow_fq12 from dory_fq12_utils
use jolt_optimizations::dory_fq12_utils::pow_fq12;

/// Generate expressions for one round of Dory protocol
fn generate_dory_round_expressions(
    state: &DoryState,
    round: usize,
    alpha: Fq,
    beta: Fq,
    gamma: Fq,
    s1_tilde: Fq,
    s2_tilde: Fq,
) -> Vec<Expression> {
    let mut expressions = Vec::new();

    // Compute the field inversions we'll need
    let alpha_inv = alpha.inverse().unwrap();
    let beta_inv = beta.inverse().unwrap();
    let gamma_inv = gamma.inverse().unwrap();

    // Update C expression
    // C' ← C + χ_i + β·D_2 + β^{-1}·D_1 + α·C_+ + α^{-1}·C_-
    let c_new = state.c
        + state.chi[round]
        + pow_fq12(&state.d2, beta)
        + pow_fq12(&state.d1, beta_inv)
        + pow_fq12(&state.c_plus, alpha)
        + pow_fq12(&state.c_minus, alpha_inv);

    expressions.push(Expression {
        name: format!("C_update_round_{}", round),
        lhs: fq12_to_poly12_coeffs(&c_new),
        rhs: vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.c),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.chi[round]),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.d2),
                exponent: beta,
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.d1),
                exponent: beta_inv,
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.c_plus),
                exponent: alpha,
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.c_minus),
                exponent: alpha_inv,
            },
        ],
        quotient: None,
    });

    // Update D1 expression
    // D_1' ← α·D_{1L} + D_{1R} + αβ·Δ_{1L} + β·Δ_{1R}
    let d1_new = pow_fq12(&state.d1l, alpha)
        + state.d1r
        + pow_fq12(&state.delta_1l, alpha * beta)
        + pow_fq12(&state.delta_1r, beta);

    expressions.push(Expression {
        name: format!("D1_update_round_{}", round),
        lhs: fq12_to_poly12_coeffs(&d1_new),
        rhs: vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.d1l),
                exponent: alpha,
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.d1r),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.delta_1l),
                exponent: alpha * beta,
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.delta_1r),
                exponent: beta,
            },
        ],
        quotient: None,
    });

    // Update D2 expression
    // D_2' ← α^{-1}·D_{2L} + D_{2R} + α^{-1}β^{-1}·Δ_{2L} + β^{-1}·Δ_{2R}
    let d2_new = pow_fq12(&state.d2l, alpha_inv)
        + state.d2r
        + pow_fq12(&state.delta_2l, alpha_inv * beta_inv)
        + pow_fq12(&state.delta_2r, beta_inv);

    expressions.push(Expression {
        name: format!("D2_update_round_{}", round),
        lhs: fq12_to_poly12_coeffs(&d2_new),
        rhs: vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.d2l),
                exponent: alpha_inv,
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.d2r),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.delta_2l),
                exponent: alpha_inv * beta_inv,
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.delta_2r),
                exponent: beta_inv,
            },
        ],
        quotient: None,
    });

    // Fold Scalars - C expression
    // C' ← C + s̃_1·s̃_2·H_T + γ·e(H_1, E_2) + γ^{-1}·e(E_1, H_2)
    let c_fold = state.c
        + pow_fq12(&state.h_t, s1_tilde * s2_tilde)
        + pow_fq12(&state.e_h1_e2, gamma)
        + pow_fq12(&state.e_e1_h2, gamma_inv);

    expressions.push(Expression {
        name: format!("C_fold_round_{}", round),
        lhs: fq12_to_poly12_coeffs(&c_fold),
        rhs: vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.c),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.h_t),
                exponent: s1_tilde * s2_tilde,
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.e_h1_e2),
                exponent: gamma,
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.e_e1_h2),
                exponent: gamma_inv,
            },
        ],
        quotient: None,
    });

    // Fold Scalars - D1 expression
    // D_1' ← D_1 + e(H_1, Γ_{2,0}·s̃_1·γ)
    let d1_fold = state.d1 + pow_fq12(&state.e_h1_gamma2, s1_tilde * gamma);

    expressions.push(Expression {
        name: format!("D1_fold_round_{}", round),
        lhs: fq12_to_poly12_coeffs(&d1_fold),
        rhs: vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.d1),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.e_h1_gamma2),
                exponent: s1_tilde * gamma,
            },
        ],
        quotient: None,
    });

    // Fold Scalars - D2 expression
    // D_2' ← D_2 + e(Γ_{1,0}·s̃_2·γ^{-1}, H_2)
    let d2_fold = state.d2 + pow_fq12(&state.e_gamma1_h2, s2_tilde * gamma_inv);

    expressions.push(Expression {
        name: format!("D2_fold_round_{}", round),
        lhs: fq12_to_poly12_coeffs(&d2_fold),
        rhs: vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.d2),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.e_gamma1_h2),
                exponent: s2_tilde * gamma_inv,
            },
        ],
        quotient: None,
    });

    expressions
}

/// Compute Dory expressions naively in Fq12
fn compute_dory_naive(
    state: &DoryState,
    num_rounds: usize,
    alphas: &[Fq],
    betas: &[Fq],
    gammas: &[Fq],
    s1_tildes: &[Fq],
    s2_tildes: &[Fq],
) -> (Fq12, Fq12, Fq12) {
    let mut c = state.c;
    let mut d1 = state.d1;
    let mut d2 = state.d2;

    for round in 0..num_rounds {
        let alpha = alphas[round];
        let beta = betas[round];
        let gamma = gammas[round];
        let s1_tilde = s1_tildes[round];
        let s2_tilde = s2_tildes[round];

        let alpha_inv = alpha.inverse().unwrap();
        let beta_inv = beta.inverse().unwrap();
        let gamma_inv = gamma.inverse().unwrap();

        // Update C
        c = c
            + state.chi[round]
            + pow_fq12(&state.d2, beta)
            + pow_fq12(&state.d1, beta_inv)
            + pow_fq12(&state.c_plus, alpha)
            + pow_fq12(&state.c_minus, alpha_inv);

        // Update D1
        d1 = pow_fq12(&state.d1l, alpha)
            + state.d1r
            + pow_fq12(&state.delta_1l, alpha * beta)
            + pow_fq12(&state.delta_1r, beta);

        // Update D2
        d2 = pow_fq12(&state.d2l, alpha_inv)
            + state.d2r
            + pow_fq12(&state.delta_2l, alpha_inv * beta_inv)
            + pow_fq12(&state.delta_2r, beta_inv);

        // Fold scalars
        c = c
            + pow_fq12(&state.h_t, s1_tilde * s2_tilde)
            + pow_fq12(&state.e_h1_e2, gamma)
            + pow_fq12(&state.e_e1_h2, gamma_inv);

        d1 = d1 + pow_fq12(&state.e_h1_gamma2, s1_tilde * gamma);
        d2 = d2 + pow_fq12(&state.e_gamma1_h2, s2_tilde * gamma_inv);
    }

    (c, d1, d2)
}

fn bench_dory_verification(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut group = c.benchmark_group("dory_verification");
    println!("benching...");
    // Test different numbers of rounds
    for num_rounds in [15].iter() {
        let state = DoryState::random(*num_rounds);

        // Generate random challenges for each round
        let alphas: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();
        let betas: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();
        let gammas: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();
        let s1_tildes: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();
        let s2_tildes: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();

        // Generate all expressions for all rounds
        println!("Generating expressions for {} rounds...", num_rounds);
        let mut all_expressions = Vec::new();
        for round in 0..*num_rounds {
            println!("  Generating expressions for round {}...", round);
            let round_expressions = generate_dory_round_expressions(
                &state,
                round,
                alphas[round],
                betas[round],
                gammas[round],
                s1_tildes[round],
                s2_tildes[round],
            );
            all_expressions.extend(round_expressions);
        }
        println!("Total expressions generated: {}", all_expressions.len());

        // For benchmarking, use dummy quotients since real computation requires Fq12 values
        // In practice, the prover would compute these from the actual Fq12 elements
        println!("Setting up quotients for benchmarking...");
        let g = g_coeffs();
        let mut expressions_with_quotients = all_expressions.clone();
        for expression in expressions_with_quotients.iter_mut() {
            // Use a dummy quotient for benchmarking
            // Real implementation would compute from Fq12 values
            expression.quotient = Some(vec![Fq::one(); 10]);
        }
        println!("Quotients ready!");

        // Benchmark naive computation (verifier computes everything in Fq12)
        group.bench_with_input(
            BenchmarkId::new("naive_fq12", num_rounds),
            num_rounds,
            |b, _| {
                b.iter(|| {
                    black_box(compute_dory_naive(
                        &state,
                        *num_rounds,
                        &alphas,
                        &betas,
                        &gammas,
                        &s1_tildes,
                        &s2_tildes,
                    ))
                })
            },
        );

        // Benchmark quotient-based verification (verifier just checks)
        let r = Fq::rand(&mut rng);
        let batch_gammas: Vec<Fq> = (0..expressions_with_quotients.len())
            .map(|_| Fq::rand(&mut rng))
            .collect();
        let g_array: [Fq; 13] = g.clone().try_into().unwrap();

        group.bench_with_input(
            BenchmarkId::new("quotient_based", num_rounds),
            num_rounds,
            |b, _| {
                b.iter(|| {
                    black_box(verify_batched_expressions(
                        &expressions_with_quotients,
                        r,
                        &batch_gammas,
                        &g_array,
                    ))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dory_verification);
criterion_main!(benches);
