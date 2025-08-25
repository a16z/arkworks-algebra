use ark_bn254::{Fq, Fq12};
use ark_ff::{Field, One, UniformRand, Zero};
use ark_std::test_rng;
use jolt_optimizations::{
    batched_expressions::{
        attach_quotient, compute_residual_no_reduce, compute_residual_from_fq12, expression_from_fq12, prepare_round_quotients,
        quotient_divide_by_g, verify_batched_expressions, BatchCheckResult, Expression,
        ExpressionTerm, Poly12, RoundSpec,
    },
    eval_poly12, eval_poly_vec, fq12_to_poly12_coeffs, g_coeffs, g_eval, poly_div_rem_monic,
    poly_mul,
};

mod test_helpers;

#[test]
fn test_simple_multiplication_expression() {
    let mut rng = test_rng();
    
    // Create a simple expression: a * b = c
    let a = Fq12::rand(&mut rng);
    let b = Fq12::rand(&mut rng);
    let c = a * b;
    
    // Convert to polynomial form
    let c_poly = fq12_to_poly12_coeffs(&c);
    let a_poly = fq12_to_poly12_coeffs(&a);
    let b_poly = fq12_to_poly12_coeffs(&b);
    
    // Create expression: c(X) = a(X) * b(X) mod g(X)
    let mut expression = Expression {
        name: "a*b=c".to_string(),
        lhs: c_poly,
        rhs: vec![
            ExpressionTerm {
                poly: a_poly,
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: b_poly,
                exponent: Fq::one(),
            },
        ],
        quotient: None,
    };
    
    // Compute quotient
    let g = g_coeffs();
    let residual = compute_residual_no_reduce(&expression.lhs, &expression.rhs);
    let quotient = quotient_divide_by_g(residual, &g);
    attach_quotient(&mut expression, quotient);
    
    // Verify at random point
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];
    let g_array: [Fq; 13] = g.try_into().unwrap();
    
    let result = verify_batched_expressions(&[expression], r, &gammas, &g_array);
    assert!(result.ok, "Simple multiplication expression should verify");
}

#[test]
fn test_exponentiation_expression() {
    let mut rng = test_rng();
    
    // Create expression: a^3 = c
    let a = Fq12::rand(&mut rng);
    let c = a * a * a;
    
    // Use the proper method for expressions with exponents
    let expression = test_helpers::create_test_expression_with_quotient(
        "a^3=c".to_string(),
        c,
        vec![(a, Fq::from(3u64))],
    );
    
    // Verify
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];
    let g = g_coeffs();
    let g_array: [Fq; 13] = g.try_into().unwrap();
    
    let result = verify_batched_expressions(&[expression], r, &gammas, &g_array);
    assert!(result.ok, "Exponentiation expression should verify");
}

#[test]
fn test_multi_exponentiation_expression() {
    let mut rng = test_rng();
    
    // Create expression: a^2 * b^3 * c = d
    let a = Fq12::rand(&mut rng);
    let b = Fq12::rand(&mut rng);
    let c = Fq12::rand(&mut rng);
    let d = a.square() * b * b * b * c;
    
    // Use the proper method for expressions with exponents
    let expression = test_helpers::create_test_expression_with_quotient(
        "a^2*b^3*c=d".to_string(),
        d,
        vec![(a, Fq::from(2u64)), (b, Fq::from(3u64)), (c, Fq::one())],
    );
    
    // Verify
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];
    let g = g_coeffs();
    let g_array: [Fq; 13] = g.try_into().unwrap();
    
    let result = verify_batched_expressions(&[expression], r, &gammas, &g_array);
    assert!(result.ok, "Multi-exponentiation expression should verify");
}

#[test]
fn test_batched_expressions() {
    let mut rng = test_rng();
    
    // Create multiple expressions
    let mut expressions = Vec::new();
    
    // Expression 1: a * b = c
    let a1 = Fq12::rand(&mut rng);
    let b1 = Fq12::rand(&mut rng);
    let c1 = a1 * b1;
    
    expressions.push(test_helpers::create_test_expression_with_quotient(
        "c1=a1*b1".to_string(),
        c1,
        vec![(a1, Fq::one()), (b1, Fq::one())],
    ));
    
    // Expression 2: d^2 = e
    let d2 = Fq12::rand(&mut rng);
    let e2 = d2.square();
    
    expressions.push(test_helpers::create_test_expression_with_quotient(
        "e2=d2^2".to_string(),
        e2,
        vec![(d2, Fq::from(2u64))],
    ));
    
    // Expression 3: f * g^3 = h
    let f3 = Fq12::rand(&mut rng);
    let g3 = Fq12::rand(&mut rng);
    let h3 = f3 * g3 * g3 * g3;
    
    expressions.push(test_helpers::create_test_expression_with_quotient(
        "h3=f3*g3^3".to_string(),
        h3,
        vec![(f3, Fq::one()), (g3, Fq::from(3u64))],
    ));
    
    // No need to prepare quotients - already done
    let round = RoundSpec {
        expressions,
        transcript_bytes: None,
    };
    
    // Verify with random gammas
    let r = Fq::rand(&mut rng);
    let gammas = vec![
        Fq::rand(&mut rng),
        Fq::rand(&mut rng),
        Fq::rand(&mut rng),
    ];
    let g = g_coeffs();
    let g_array: [Fq; 13] = g.try_into().unwrap();
    
    let result = verify_batched_expressions(&round.expressions, r, &gammas, &g_array);
    assert!(result.ok, "Batched expressions should verify");
}

#[test]
fn test_zero_exponent() {
    let mut rng = test_rng();
    
    // Create expression: a^0 = 1 (should be 1 in Fq12)
    let a = Fq12::rand(&mut rng);
    let one = Fq12::one();
    
    let one_poly = fq12_to_poly12_coeffs(&one);
    let a_poly = fq12_to_poly12_coeffs(&a);
    
    let mut expression = Expression {
        name: "a^0=1".to_string(),
        lhs: one_poly,
        rhs: vec![ExpressionTerm {
            poly: a_poly,
            exponent: Fq::zero(),
        }],
        quotient: None,
    };
    
    // Compute quotient
    let g = g_coeffs();
    let residual = compute_residual_no_reduce(&expression.lhs, &expression.rhs);
    let quotient = quotient_divide_by_g(residual, &g);
    attach_quotient(&mut expression, quotient);
    
    // Verify
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];
    let g_array: [Fq; 13] = g.try_into().unwrap();
    
    let result = verify_batched_expressions(&[expression], r, &gammas, &g_array);
    assert!(result.ok, "Zero exponent expression should verify");
}

#[test]
fn test_rejection_of_incorrect_expression() {
    let mut rng = test_rng();
    
    // Create an INCORRECT expression: a * b = c + 1
    let a = Fq12::rand(&mut rng);
    let b = Fq12::rand(&mut rng);
    let c = a * b;
    let c_bad = c + Fq12::one();
    
    let c_bad_poly = fq12_to_poly12_coeffs(&c_bad);
    let a_poly = fq12_to_poly12_coeffs(&a);
    let b_poly = fq12_to_poly12_coeffs(&b);
    
    // Create the correct quotient for a * b (not c_bad)
    let c_poly = fq12_to_poly12_coeffs(&c);
    let mut correct_expression = Expression {
        name: "correct".to_string(),
        lhs: c_poly,
        rhs: vec![
            ExpressionTerm {
                poly: a_poly,
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: b_poly,
                exponent: Fq::one(),
            },
        ],
        quotient: None,
    };
    
    let g = g_coeffs();
    let residual = compute_residual_no_reduce(&correct_expression.lhs, &correct_expression.rhs);
    let quotient = quotient_divide_by_g(residual, &g);
    
    // Now create bad expression with wrong lhs but correct quotient
    let bad_expression = Expression {
        name: "bad".to_string(),
        lhs: c_bad_poly,
        rhs: vec![
            ExpressionTerm {
                poly: a_poly,
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: b_poly,
                exponent: Fq::one(),
            },
        ],
        quotient: Some(quotient),
    };
    
    // Verify - should fail
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];
    let g_array: [Fq; 13] = g.try_into().unwrap();
    
    let result = verify_batched_expressions(&[bad_expression], r, &gammas, &g_array);
    assert!(!result.ok, "Incorrect expression should not verify");
}

#[test]
fn test_randomized_expressions() {
    let mut rng = test_rng();
    
    // Run multiple random trials
    for _ in 0..100 {
        // Generate random Fq12 elements
        let a = Fq12::rand(&mut rng);
        let b = Fq12::rand(&mut rng);
        let c = Fq12::rand(&mut rng);
        
        // Compute various products
        let ab = a * b;
        let bc = b * c;
        let abc = a * b * c;
        let a2b = a.square() * b;
        let b3 = b * b * b;
        
        // Create multiple expressions
        let expressions = vec![
            test_helpers::create_test_expression_with_quotient(
                "ab=a*b".to_string(),
                ab,
                vec![(a, Fq::one()), (b, Fq::one())],
            ),
            test_helpers::create_test_expression_with_quotient(
                "bc=b*c".to_string(),
                bc,
                vec![(b, Fq::one()), (c, Fq::one())],
            ),
            test_helpers::create_test_expression_with_quotient(
                "abc=a*b*c".to_string(),
                abc,
                vec![(a, Fq::one()), (b, Fq::one()), (c, Fq::one())],
            ),
            test_helpers::create_test_expression_with_quotient(
                "a2b=a^2*b".to_string(),
                a2b,
                vec![(a, Fq::from(2u64)), (b, Fq::one())],
            ),
            test_helpers::create_test_expression_with_quotient("b3=b^3".to_string(), b3, vec![(b, Fq::from(3u64))]),
        ];
        
        // No need to prepare quotients - already done
        let round = RoundSpec {
            expressions,
            transcript_bytes: None,
        };
        
        // Verify with random challenge and gammas
        let r = Fq::rand(&mut rng);
        let gammas: Vec<Fq> = (0..5).map(|_| Fq::rand(&mut rng)).collect();
        let g = g_coeffs();
        let g_array: [Fq; 13] = g.try_into().unwrap();
        
        let result = verify_batched_expressions(&round.expressions, r, &gammas, &g_array);
        assert!(
            result.ok,
            "Random expressions should verify (lhs={:?}, rhs={:?})",
            result.lhs, result.rhs
        );
    }
}

#[test]
fn test_large_exponents() {
    let mut rng = test_rng();
    
    // Test with larger exponents
    let a = Fq12::rand(&mut rng);
    let mut result = Fq12::one();
    let exp = 17u64;
    
    for _ in 0..exp {
        result *= a;
    }
    
    let expression = test_helpers::create_test_expression_with_quotient(
        "a^17".to_string(),
        result,
        vec![(a, Fq::from(exp))],
    );
    
    // Verify
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];
    let g = g_coeffs();
    let g_array: [Fq; 13] = g.try_into().unwrap();
    
    let result = verify_batched_expressions(&[expression], r, &gammas, &g_array);
    assert!(result.ok, "Large exponent expression should verify");
}

#[test]
fn test_empty_rhs() {
    let mut rng = test_rng();
    
    // Create expression: 1 = (empty product)
    let one = Fq12::one();
    let one_poly = fq12_to_poly12_coeffs(&one);
    
    let mut expression = Expression {
        name: "1=empty".to_string(),
        lhs: one_poly,
        rhs: vec![], // Empty RHS should be treated as 1
        quotient: None,
    };
    
    // Compute quotient
    let g = g_coeffs();
    let residual = compute_residual_no_reduce(&expression.lhs, &expression.rhs);
    let quotient = quotient_divide_by_g(residual, &g);
    attach_quotient(&mut expression, quotient);
    
    // Verify
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];
    let g_array: [Fq; 13] = g.try_into().unwrap();
    
    let result = verify_batched_expressions(&[expression], r, &gammas, &g_array);
    assert!(result.ok, "Empty RHS expression should verify");
}

#[test]
fn test_direct_gt_comparison() {
    let mut rng = test_rng();
    
    // Test that our polynomial approach matches direct GT computation
    let a = Fq12::rand(&mut rng);
    let b = Fq12::rand(&mut rng);
    let c_direct = a * b; // Direct GT multiplication
    
    // Polynomial approach
    let a_poly = fq12_to_poly12_coeffs(&a);
    let b_poly = fq12_to_poly12_coeffs(&b);
    let prod = poly_mul(&a_poly[..], &b_poly[..]);
    
    let g = g_coeffs();
    let (_q, rem) = poly_div_rem_monic(prod, &g);
    
    // Remainder should represent c
    let mut rem_padded = rem;
    rem_padded.resize(12, Fq::zero());
    let c_poly = fq12_to_poly12_coeffs(&c_direct);
    
    assert_eq!(
        rem_padded,
        c_poly.to_vec(),
        "Polynomial remainder should match direct GT computation"
    );
}