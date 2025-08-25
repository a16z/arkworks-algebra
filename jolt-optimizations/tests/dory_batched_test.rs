use ark_bn254::{Fq, Fq12};
use ark_ff::{Field, One, UniformRand};
use ark_std::test_rng;

use jolt_optimizations::batched_expressions::verify_batched_expressions;
use jolt_optimizations::dory_fq12_utils::{pow_fq12, DoryState};
use jolt_optimizations::{fq12_to_poly12_coeffs, g_coeffs};

mod test_helpers;

use jolt_optimizations::batched_expressions::Expression;

/// Generate Dory expressions with proper polynomial quotients for testing
fn generate_test_dory_expressions(
    state: &DoryState,
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
    let c_new = state.compute_c_update(round, alpha, beta);
    expressions.push(test_helpers::create_test_expression_with_quotient(
        format!("C_update_round_{}", round),
        c_new,
        vec![
            (state.c, Fq::one()),
            (state.chi[round], Fq::one()),
            (state.d2, beta),
            (state.d1, beta_inv),
            (state.c_plus, alpha),
            (state.c_minus, alpha_inv),
        ],
    ));
    
    // D1 update expression
    let d1_new = state.compute_d1_update(alpha, beta);
    expressions.push(test_helpers::create_test_expression_with_quotient(
        format!("D1_update_round_{}", round),
        d1_new,
        vec![
            (state.d1l, alpha),
            (state.d1r, Fq::one()),
            (state.delta_1l, alpha * beta),
            (state.delta_1r, beta),
        ],
    ));
    
    // D2 update expression
    let d2_new = state.compute_d2_update(alpha, beta);
    expressions.push(test_helpers::create_test_expression_with_quotient(
        format!("D2_update_round_{}", round),
        d2_new,
        vec![
            (state.d2l, alpha_inv),
            (state.d2r, Fq::one()),
            (state.delta_2l, alpha_inv * beta_inv),
            (state.delta_2r, beta_inv),
        ],
    ));
    
    // C fold expression
    let c_fold = state.compute_c_fold(gamma, s1_tilde, s2_tilde);
    expressions.push(test_helpers::create_test_expression_with_quotient(
        format!("C_fold_round_{}", round),
        c_fold,
        vec![
            (state.c, Fq::one()),
            (state.h_t, s1_tilde * s2_tilde),
            (state.e_h1_e2, gamma),
            (state.e_e1_h2, gamma_inv),
        ],
    ));
    
    // D1 fold expression
    let d1_fold = state.compute_d1_fold(gamma, s1_tilde);
    expressions.push(test_helpers::create_test_expression_with_quotient(
        format!("D1_fold_round_{}", round),
        d1_fold,
        vec![
            (state.d1, Fq::one()),
            (state.e_h1_gamma2, s1_tilde * gamma),
        ],
    ));
    
    // D2 fold expression
    let d2_fold = state.compute_d2_fold(gamma, s2_tilde);
    expressions.push(test_helpers::create_test_expression_with_quotient(
        format!("D2_fold_round_{}", round),
        d2_fold,
        vec![
            (state.d2, Fq::one()),
            (state.e_gamma1_h2, s2_tilde * gamma_inv),
        ],
    ));
    
    expressions
}

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
    let expressions = generate_test_dory_expressions(
        &state, 0, alpha, beta, gamma, s1_tilde, s2_tilde,
    );

    // Verify at random point
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one(); expressions.len()];
    let g = g_coeffs();
    let g_array: [Fq; 13] = g.try_into().unwrap();

    let result = verify_batched_expressions(&expressions, r, &gammas, &g_array);
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

        let round_expressions = generate_test_dory_expressions(
            &state,
            round, alpha, beta, gamma, s1_tilde, s2_tilde,
        );
        all_expressions.extend(round_expressions);
    }

    // Verify all expressions together
    let r = Fq::rand(&mut rng);
    let gammas: Vec<Fq> = (0..all_expressions.len())
        .map(|_| Fq::rand(&mut rng))
        .collect();
    let g = g_coeffs();
    let g_array: [Fq; 13] = g.try_into().unwrap();

    let result = verify_batched_expressions(&all_expressions, r, &gammas, &g_array);
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
    let mut expressions = state.generate_round_expressions(
        0, alpha, beta, gamma, s1_tilde, s2_tilde,
    );

    // Tamper with the first expression's LHS
    let tampered_c = state.c + Fq12::one(); // Add 1 to C
    let tampered_expression = test_helpers::create_test_expression_with_quotient(
        "tampered_C".to_string(),
        tampered_c,
        vec![
            (state.c, Fq::one()),
            (state.chi[0], Fq::one()),
            (state.d2, beta),
            (state.d1, beta.inverse().unwrap()),
            (state.c_plus, alpha),
            (state.c_minus, alpha.inverse().unwrap()),
        ],
    );

    // Replace the first expression with the tampered one
    expressions[0] = tampered_expression;

    // Verify should fail
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one(); expressions.len()];
    let g = g_coeffs();
    let g_array: [Fq; 13] = g.try_into().unwrap();

    let result = verify_batched_expressions(&expressions, r, &gammas, &g_array);
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
    let wrong_expression = test_helpers::create_test_expression_with_quotient(
        "D1_wrong_exp".to_string(),
        d1_correct,
        vec![
            (state.d1l, alpha + Fq::one()), // Wrong exponent!
            (state.d1r, Fq::one()),
            (state.delta_1l, alpha * beta),
            (state.delta_1r, beta),
        ],
    );

    // Verify should fail
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];
    let g = g_coeffs();
    let g_array: [Fq; 13] = g.try_into().unwrap();

    let result = verify_batched_expressions(&[wrong_expression], r, &gammas, &g_array);
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
    let gamma_inv = gamma.inverse().unwrap();

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
    let expressions = state.generate_round_expressions(
        0, alpha, beta, gamma, s1_tilde, s2_tilde,
    );

    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one(); expressions.len()];
    let g = g_coeffs();
    let g_array: [Fq; 13] = g.try_into().unwrap();

    let result = verify_batched_expressions(&expressions, r, &gammas, &g_array);
    assert!(result.ok, "Batched verification should succeed");
}

#[test]
fn test_dory_zero_exponent() {
    let mut rng = test_rng();

    // Initialize Dory state
    let state = DoryState::random(1);

    // Create a expression with zero exponent (should contribute 1)
    let result = state.c + Fq12::one(); // c + 1

    let expression = test_helpers::create_test_expression_with_quotient(
        "zero_exp_test".to_string(),
        result,
        vec![
            (state.c, Fq::one()),
            (state.d1, Fq::from(0u64)), // Zero exponent - should contribute 1
        ],
    );

    // Verify
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];
    let g = g_coeffs();
    let g_array: [Fq; 13] = g.try_into().unwrap();

    let result = verify_batched_expressions(&[expression], r, &gammas, &g_array);
    assert!(result.ok, "Zero exponent expression should verify");
}

#[test]
fn test_dory_large_batch() {
    let mut rng = test_rng();
    let num_rounds = 10;

    // Initialize Dory state
    let state = DoryState::random(num_rounds);

    // Generate expressions for all rounds
    let mut all_expressions = Vec::new();

    for round in 0..num_rounds {
        let alpha = Fq::rand(&mut rng);
        let beta = Fq::rand(&mut rng);
        let gamma = Fq::rand(&mut rng);
        let s1_tilde = Fq::rand(&mut rng);
        let s2_tilde = Fq::rand(&mut rng);

        let round_expressions = generate_test_dory_expressions(
            &state,
            round, alpha, beta, gamma, s1_tilde, s2_tilde,
        );
        all_expressions.extend(round_expressions);
    }

    println!("Testing batch of {} expressions", all_expressions.len());

    // Verify with random linear combination
    let r = Fq::rand(&mut rng);
    let gammas: Vec<Fq> = (0..all_expressions.len())
        .map(|_| Fq::rand(&mut rng))
        .collect();
    let g = g_coeffs();
    let g_array: [Fq; 13] = g.try_into().unwrap();

    let result = verify_batched_expressions(&all_expressions, r, &gammas, &g_array);
    assert!(result.ok, "Large batch should verify");
}
