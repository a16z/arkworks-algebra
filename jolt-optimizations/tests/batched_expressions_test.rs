use ark_bn254::{Fq, Fq12};
use ark_ff::{Field, One, UniformRand, Zero};
use ark_std::test_rng;
use jolt_optimizations::eval_poly12;
use jolt_optimizations::eval_poly_vec;
use jolt_optimizations::g_eval;
use jolt_optimizations::{
    batched_expressions::{
        verify_batched_expressions, Expression, ExpressionTerm, RoundExpresions,
    },
    fq12_to_poly12_coeffs, g_coeffs, poly_div_rem_monic, poly_mul,
};

#[test]
fn test_simple_multiplication_expression() {
    let mut rng = test_rng();

    // Create a simple expression: a * b = c
    let a = Fq12::rand(&mut rng);
    let b = Fq12::rand(&mut rng);
    let c = a * b;

    // Convert to polynomials
    let a_poly = fq12_to_poly12_coeffs(&a);
    let b_poly = fq12_to_poly12_coeffs(&b);
    let c_poly = fq12_to_poly12_coeffs(&c);

    // Create expression: c(X) = a(X) * b(X) mod g(X)
    let expression = Expression::new(
        "a*b=c".to_string(),
        c_poly,
        vec![
            ExpressionTerm {
                poly: a_poly,
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: b_poly,
                exponent: Fq::one(),
            },
        ],
    );

    // Verify at random point
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];

    let result = verify_batched_expressions(&[expression], r, &gammas);
    assert!(result.ok, "Simple multiplication expression should verify");
}

#[test]
fn test_exponentiation_expression() {
    let mut rng = test_rng();

    // Create expression: a^3 = c
    let a = Fq12::rand(&mut rng);
    let c = a * a * a;

    // Convert to polynomials
    let a_poly = fq12_to_poly12_coeffs(&a);
    let c_poly = fq12_to_poly12_coeffs(&c);

    let expression = Expression::new(
        "a^3=c".to_string(),
        c_poly,
        vec![ExpressionTerm {
            poly: a_poly,
            exponent: Fq::from(3u64),
        }],
    );

    // Verify
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];

    let result = verify_batched_expressions(&[expression], r, &gammas);
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

    // Convert to polynomials
    let a_poly = fq12_to_poly12_coeffs(&a);
    let b_poly = fq12_to_poly12_coeffs(&b);
    let c_poly = fq12_to_poly12_coeffs(&c);
    let d_poly = fq12_to_poly12_coeffs(&d);

    let expression = Expression::new(
        "a^2*b^3*c=d".to_string(),
        d_poly,
        vec![
            ExpressionTerm {
                poly: a_poly,
                exponent: Fq::from(2u64),
            },
            ExpressionTerm {
                poly: b_poly,
                exponent: Fq::from(3u64),
            },
            ExpressionTerm {
                poly: c_poly,
                exponent: Fq::one(),
            },
        ],
    );

    // Verify
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];

    let result = verify_batched_expressions(&[expression], r, &gammas);
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

    expressions.push(Expression::new(
        "c1=a1*b1".to_string(),
        fq12_to_poly12_coeffs(&c1),
        vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&a1),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&b1),
                exponent: Fq::one(),
            },
        ],
    ));

    // Expression 2: d^2 = e
    let d2 = Fq12::rand(&mut rng);
    let e2 = d2.square();

    expressions.push(Expression::new(
        "e2=d2^2".to_string(),
        fq12_to_poly12_coeffs(&e2),
        vec![ExpressionTerm {
            poly: fq12_to_poly12_coeffs(&d2),
            exponent: Fq::from(2u64),
        }],
    ));

    // Expression 3: f * g^3 = h
    let f3 = Fq12::rand(&mut rng);
    let g3 = Fq12::rand(&mut rng);
    let h3 = f3 * g3 * g3 * g3;

    expressions.push(Expression::new(
        "h3=f3*g3^3".to_string(),
        fq12_to_poly12_coeffs(&h3),
        vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&f3),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&g3),
                exponent: Fq::from(3u64),
            },
        ],
    ));

    let round = RoundExpresions { expressions };

    // Verify with random gammas
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::rand(&mut rng), Fq::rand(&mut rng), Fq::rand(&mut rng)];

    let result = verify_batched_expressions(&round.expressions, r, &gammas);
    assert!(result.ok, "Batched expressions should verify");
}

#[test]
fn test_zero_exponent() {
    let mut rng = test_rng();

    // Create expression: a^0 = 1 (should be 1 in Fq12)
    let a = Fq12::rand(&mut rng);
    let one = Fq12::one();

    let expression = Expression::new(
        "a^0=1".to_string(),
        fq12_to_poly12_coeffs(&one),
        vec![ExpressionTerm {
            poly: fq12_to_poly12_coeffs(&a),
            exponent: Fq::zero(),
        }],
    );

    // Verify
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];

    let result = verify_batched_expressions(&[expression], r, &gammas);
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

    // Create the correct expression for a * b to get the correct quotient
    let correct_expression = Expression::new(
        "correct".to_string(),
        fq12_to_poly12_coeffs(&c),
        vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&a),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&b),
                exponent: Fq::one(),
            },
        ],
    );

    // Now create bad expression with wrong lhs but steal the correct quotient
    // (This tests that verification catches mismatched expressions)
    let bad_expression = Expression {
        name: "bad".to_string(),
        lhs: fq12_to_poly12_coeffs(&c_bad),
        rhs: vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&a),
                exponent: Fq::one(),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&b),
                exponent: Fq::one(),
            },
        ],
        quotient: correct_expression.quotient.clone(),
    };

    // Verify - should fail
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];

    let result = verify_batched_expressions(&[bad_expression], r, &gammas);
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
            Expression::new(
                "ab=a*b".to_string(),
                fq12_to_poly12_coeffs(&ab),
                vec![
                    ExpressionTerm {
                        poly: fq12_to_poly12_coeffs(&a),
                        exponent: Fq::one(),
                    },
                    ExpressionTerm {
                        poly: fq12_to_poly12_coeffs(&b),
                        exponent: Fq::one(),
                    },
                ],
            ),
            Expression::new(
                "bc=b*c".to_string(),
                fq12_to_poly12_coeffs(&bc),
                vec![
                    ExpressionTerm {
                        poly: fq12_to_poly12_coeffs(&b),
                        exponent: Fq::one(),
                    },
                    ExpressionTerm {
                        poly: fq12_to_poly12_coeffs(&c),
                        exponent: Fq::one(),
                    },
                ],
            ),
            Expression::new(
                "abc=a*b*c".to_string(),
                fq12_to_poly12_coeffs(&abc),
                vec![
                    ExpressionTerm {
                        poly: fq12_to_poly12_coeffs(&a),
                        exponent: Fq::one(),
                    },
                    ExpressionTerm {
                        poly: fq12_to_poly12_coeffs(&b),
                        exponent: Fq::one(),
                    },
                    ExpressionTerm {
                        poly: fq12_to_poly12_coeffs(&c),
                        exponent: Fq::one(),
                    },
                ],
            ),
            Expression::new(
                "a2b=a^2*b".to_string(),
                fq12_to_poly12_coeffs(&a2b),
                vec![
                    ExpressionTerm {
                        poly: fq12_to_poly12_coeffs(&a),
                        exponent: Fq::from(2u64),
                    },
                    ExpressionTerm {
                        poly: fq12_to_poly12_coeffs(&b),
                        exponent: Fq::one(),
                    },
                ],
            ),
            Expression::new(
                "b3=b^3".to_string(),
                fq12_to_poly12_coeffs(&b3),
                vec![ExpressionTerm {
                    poly: fq12_to_poly12_coeffs(&b),
                    exponent: Fq::from(3u64),
                }],
            ),
        ];

        let round = RoundExpresions { expressions };

        // Verify with random challenge and gammas
        let r = Fq::rand(&mut rng);
        let gammas: Vec<Fq> = (0..5).map(|_| Fq::rand(&mut rng)).collect();

        let result = verify_batched_expressions(&round.expressions, r, &gammas);
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

    let expression = Expression::new(
        "a^17".to_string(),
        fq12_to_poly12_coeffs(&result),
        vec![ExpressionTerm {
            poly: fq12_to_poly12_coeffs(&a),
            exponent: Fq::from(exp),
        }],
    );

    // Verify
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];

    let result = verify_batched_expressions(&[expression], r, &gammas);
    assert!(result.ok, "Large exponent expression should verify");
}

#[test]
fn test_empty_rhs() {
    let mut rng = test_rng();

    // Create expression: 1 = (empty product)
    let one = Fq12::one();

    let expression = Expression::new(
        "1=empty".to_string(),
        fq12_to_poly12_coeffs(&one),
        vec![], // Empty RHS should be treated as 1
    );

    // Verify
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];

    let result = verify_batched_expressions(&[expression], r, &gammas);
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

#[test]
fn test_single_power_a2_eq_c() {
    let mut rng = test_rng();

    let a = Fq12::rand(&mut rng);
    let c = a.square();

    let expr = Expression::new(
        "a^2=c".to_string(),
        fq12_to_poly12_coeffs(&c),
        vec![ExpressionTerm {
            poly: fq12_to_poly12_coeffs(&a),
            exponent: Fq::from(2u64),
        }],
    );

    // quotient should be zero (or empty after trimming)
    let q = expr.quotient.as_ref().unwrap();
    assert!(q.iter().all(|x| x.is_zero()), "q should be 0 for a^2=c");

    // verify via the batched checker
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one()];
    let res = verify_batched_expressions(&[expr.clone()], r, &gammas);
    assert!(res.ok, "batched verify should pass");

    // manual spot-check: lhs(r) - (a(r))^2 == q(r)*g(r) == 0
    let lhs_r = eval_poly12(&expr.lhs, &r);
    let a_r = eval_poly12(&expr.rhs[0].poly, &r);
    use ark_ff::PrimeField;
    let rhs_r = a_r.pow(&Fq::from(2u64).into_bigint());
    let q_r = eval_poly_vec(&q, &r);
    let g_r = g_eval(&r);
    assert_eq!(lhs_r - rhs_r, q_r * g_r);
}

#[test]
fn test_multifactor_batch_round() {
    let mut rng = test_rng();

    // Build a D1-type constraint: D1' = D1L^α * D1R * Δ1L^{αβ} * Δ1R^β
    let d1l = Fq12::rand(&mut rng);
    let d1r = Fq12::rand(&mut rng);
    let dl1 = Fq12::rand(&mut rng);
    let dr1 = Fq12::rand(&mut rng);

    let alpha = Fq::rand(&mut rng);
    let beta = Fq::rand(&mut rng);
    let alphab = alpha * beta;

    // form D1' in the field, to ensure a true relation
    use ark_ff::PrimeField;
    let mut d1p = d1l;
    d1p = d1p.pow(&alpha.into_bigint());
    d1p *= &d1r;
    let mut t = dl1.pow(&alphab.into_bigint());
    d1p *= &t;
    t = dr1.pow(&beta.into_bigint());
    d1p *= &t;

    // correct expression/quotient
    let expr1 = Expression::new(
        "D1".to_string(),
        fq12_to_poly12_coeffs(&d1p),
        vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&d1l),
                exponent: alpha,
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&d1r),
                exponent: Fq::from(1u64),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&dl1),
                exponent: alphab,
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&dr1),
                exponent: beta,
            },
        ],
    );

    // Add a second (independent) constraint to test batching
    let x = Fq12::rand(&mut rng);
    let y = Fq12::rand(&mut rng);
    let z = x * y;
    let expr2 = Expression::new(
        "xy=z".to_string(),
        fq12_to_poly12_coeffs(&z),
        vec![
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&x),
                exponent: Fq::from(1),
            },
            ExpressionTerm {
                poly: fq12_to_poly12_coeffs(&y),
                exponent: Fq::from(1),
            },
        ],
    );

    // One random point and random gamma for batching
    let r = Fq::rand(&mut rng);
    let gammas = vec![Fq::one(), Fq::rand(&mut rng)];

    // Batch verify
    let res = verify_batched_expressions(&[expr1.clone(), expr2.clone()], r, &gammas);
    assert!(res.ok, "batched verify should pass");

    // Optional: ensure at least one quotient is nonzero in this multifactor case
    let any_nonzero_q = [
        expr1.quotient.as_ref().unwrap(),
        expr2.quotient.as_ref().unwrap(),
    ]
    .iter()
    .any(|q| q.iter().any(|c| !c.is_zero()));
    assert!(
        any_nonzero_q,
        "expected a nonzero quotient in multifactor case"
    );
}