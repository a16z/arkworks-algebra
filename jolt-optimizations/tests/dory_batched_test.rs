use ark_bn254::{Fq, Fq12};
use ark_ff::{Field, One, UniformRand};
use ark_std::test_rng;

use jolt_optimizations::batched_expressions::{
    verify_batched_expressions, Expression, ExpressionTerm,
};
use jolt_optimizations::dory_fq12_utils::{pow_fq12, DoryState};
use jolt_optimizations::fq12_to_poly12_coeffs;

#[test]
fn test_dory_single_round_valid() {
    let mut rng = test_rng();

    // Initialize Dory state
    let state = DoryState::random(1);

    // Generate random challenges
    let alpha = Fq::rand(&mut rng);
    let beta = Fq::rand(&mut rng);
    let gamma = Fq::rand(&mut rng);
    let s1_tilde = Fq::rand(&mut rng);
    let s2_tilde = Fq::rand(&mut rng);

    // Compute Dory updates in Fq12
    let c_new = state.compute_c_update(0, alpha, beta);
    let _d1_new = state.compute_d1_update(alpha, beta);
    let _d2_new = state.compute_d2_update(alpha, beta);
    let _c_fold = state.compute_c_fold(gamma, s1_tilde, s2_tilde);
    let _d1_fold = state.compute_d1_fold(gamma, s1_tilde);
    let _d2_fold = state.compute_d2_fold(gamma, s2_tilde);

    // Generate expressions with proper quotients
    let expressions = state.generate_round_expressions(0, alpha, beta, gamma, s1_tilde, s2_tilde);

    // Verify at random point
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one(); expressions.len()];

    let result = verify_batched_expressions(&expressions, r, &gammas);
    assert!(result.ok, "Valid Dory expressions should verify");

    // Also verify that the computed values match what we expect
    let c_new_poly = fq12_to_poly12_coeffs(&c_new);
    assert_eq!(
        expressions[0].lhs, c_new_poly,
        "C update LHS should match computed value"
    );
}

#[test]
fn test_dory_multi_round_valid() {
    let mut rng = test_rng();
    let num_rounds = 3;

    // Initialize Dory state
    let state = DoryState::random(num_rounds);

    // Generate random challenges for each round
    let mut all_expressions = Vec::new();

    for round in 0..num_rounds {
        let alpha = Fq::rand(&mut rng);
        let beta = Fq::rand(&mut rng);
        let gamma = Fq::rand(&mut rng);
        let s1_tilde = Fq::rand(&mut rng);
        let s2_tilde = Fq::rand(&mut rng);

        let round_expressions =
            state.generate_round_expressions(round, alpha, beta, gamma, s1_tilde, s2_tilde);
        all_expressions.extend(round_expressions);
    }

    // Verify all expressions together
    let r = Fq::rand(&mut rng);
    let gammas: Vec<Fq> = (0..all_expressions.len())
        .map(|_| Fq::rand(&mut rng))
        .collect();

    let result = verify_batched_expressions(&all_expressions, r, &gammas);
    assert!(
        result.ok,
        "Valid multi-round Dory expressions should verify (lhs={:?}, rhs={:?})",
        result.lhs, result.rhs
    );
}

#[test]
fn test_dory_tampering_detection() {
    let mut rng = test_rng();

    // Initialize Dory state
    let state = DoryState::random(1);

    // Generate random challenges
    let alpha = Fq::rand(&mut rng);
    let beta = Fq::rand(&mut rng);
    let gamma = Fq::rand(&mut rng);
    let s1_tilde = Fq::rand(&mut rng);
    let s2_tilde = Fq::rand(&mut rng);

    // Generate valid expressions
    let mut expressions =
        state.generate_round_expressions(0, alpha, beta, gamma, s1_tilde, s2_tilde);

    // Tamper with the first expression's LHS
    let tampered_c = state.c + Fq12::one(); // Add 1 to C
    let tampered_expression = Expression::new(
        "tampered_C".to_string(),
        fq12_to_poly12_coeffs(&tampered_c),
        vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.c),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.chi[0]),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.d2),
                exponent: beta,
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.d1),
                exponent: beta.inverse().unwrap(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.c_plus),
                exponent: alpha,
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.c_minus),
                exponent: alpha.inverse().unwrap(),
            },
        ],
    );

    // Replace the first expression with the tampered one
    expressions[0] = tampered_expression;

    // Verify should fail
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one(); expressions.len()];

    let result = verify_batched_expressions(&expressions, r, &gammas);
    assert!(!result.ok, "Tampered Dory expressions should not verify");
}

#[test]
fn test_dory_wrong_exponent_detection() {
    let mut rng = test_rng();

    // Initialize Dory state
    let state = DoryState::random(1);

    // Generate random challenges
    let alpha = Fq::rand(&mut rng);
    let beta = Fq::rand(&mut rng);

    // Compute the correct D1 update
    let d1_correct = state.compute_d1_update(alpha, beta);

    // Create a expression with wrong exponent
    let wrong_expression = Expression::new(
        "D1_wrong_exp".to_string(),
        fq12_to_poly12_coeffs(&d1_correct),
        vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.d1l),
                exponent: alpha + Fq::one(),
            }, // Wrong exponent!
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
    );

    // Verify should fail
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];

    let result = verify_batched_expressions(&[wrong_expression], r, &gammas);
    assert!(!result.ok, "Wrong exponent should not verify");
}

#[test]
fn test_dory_naive_vs_batched() {
    let mut rng = test_rng();

    // Initialize Dory state
    let state = DoryState::random(1);

    // Generate random challenges
    let alpha = Fq::rand(&mut rng);
    let beta = Fq::rand(&mut rng);
    let gamma = Fq::rand(&mut rng);
    let s1_tilde = Fq::rand(&mut rng);
    let s2_tilde = Fq::rand(&mut rng);

    // Compute updates naively in Fq12
    let alpha_inv = alpha.inverse().unwrap();
    let beta_inv = beta.inverse().unwrap();
    let _gamma_inv = gamma.inverse().unwrap();

    // Naive C update
    let c_naive = state.c
        + state.chi[0]
        + pow_fq12(&state.d2, beta)
        + pow_fq12(&state.d1, beta_inv)
        + pow_fq12(&state.c_plus, alpha)
        + pow_fq12(&state.c_minus, alpha_inv);

    // Batched approach
    let c_batched = state.compute_c_update(0, alpha, beta);

    // Should be identical
    assert_eq!(
        c_naive, c_batched,
        "Naive and batched computation should match"
    );

    // Generate expressions and verify
    let expressions = state.generate_round_expressions(0, alpha, beta, gamma, s1_tilde, s2_tilde);

    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one(); expressions.len()];

    let result = verify_batched_expressions(&expressions, r, &gammas);
    assert!(result.ok, "Batched verification should succeed");
}

#[test]
fn test_dory_zero_exponent() {
    let mut rng = test_rng();

    // Initialize Dory state
    let state = DoryState::random(1);

    // Create a expression with zero exponent (should contribute 1)
    let result = state.c + Fq12::one(); // c + 1

    let expression = Expression::new(
        "zero_exp_test".to_string(),
        fq12_to_poly12_coeffs(&result),
        vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.c),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&state.d1),
                exponent: Fq::from(0u64),
            }, // Zero exponent - should contribute 1
        ],
    );

    // Verify
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];

    let result = verify_batched_expressions(&[expression], r, &gammas);
    assert!(result.ok, "Zero exponent expression should verify");
}
