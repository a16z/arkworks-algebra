use ark_bn254::{Fq, Fq12};
use ark_ff::{Field, One, UniformRand, Zero};
use ark_std::test_rng;
use jolt_optimizations::{
    eval_poly12, eval_poly_vec, fq12_to_poly12_coeffs, g_coeffs, g_eval, poly_div_rem_monic,
    poly_mul,
};

#[test]
fn test_fq12_to_poly_identity() {
    // Test that the identity element maps to the polynomial 1
    let one = Fq12::one();
    let coeffs = fq12_to_poly12_coeffs(&one);
    
    assert_eq!(coeffs[0], Fq::one());
    for i in 1..12 {
        assert_eq!(coeffs[i], Fq::zero());
    }
}

#[test]
fn test_fq12_to_poly_zero() {
    // Test that zero maps to the zero polynomial
    let zero = Fq12::zero();
    let coeffs = fq12_to_poly12_coeffs(&zero);
    
    for i in 0..12 {
        assert_eq!(coeffs[i], Fq::zero());
    }
}

#[test]
fn test_polynomial_multiplication_mod_g() {
    let mut rng = test_rng();
    
    // Random a,b in Fq12; define c = a*b in the field (i.e., mod g)
    let a = Fq12::rand(&mut rng);
    let b = Fq12::rand(&mut rng);
    let c = a * b;
    
    // Flatten into polynomials over Fq (degree ≤ 11)
    let a12 = fq12_to_poly12_coeffs(&a);
    let b12 = fq12_to_poly12_coeffs(&b);
    let c12 = fq12_to_poly12_coeffs(&c);
    
    // Multiply the polynomials a(X), b(X) in Fq[X] without reduction
    let prod = poly_mul(&a12, &b12);
    
    // Get g(X) coefficients
    let g = g_coeffs();
    
    // Divide prod by g to get (q, rem)
    let (_q, rem) = poly_div_rem_monic(prod, &g);
    
    // Remainder should equal c12
    let mut rem_padded = rem;
    rem_padded.resize(12, Fq::zero());
    
    assert_eq!(rem_padded, c12.to_vec());
}

#[test]
fn test_schwartz_zippel_check() {
    let mut rng = test_rng();
    
    // Random a,b in Fq12
    let a = Fq12::rand(&mut rng);
    let b = Fq12::rand(&mut rng);
    let c = a * b;
    
    // Flatten to polynomials
    let a12 = fq12_to_poly12_coeffs(&a);
    let b12 = fq12_to_poly12_coeffs(&b);
    let c12 = fq12_to_poly12_coeffs(&c);
    
    // Polynomial multiplication
    let prod = poly_mul(&a12, &b12);
    let g = g_coeffs();
    let (q, _rem) = poly_div_rem_monic(prod, &g);
    
    // Random evaluation point
    let r = Fq::rand(&mut rng);
    
    // Check: a(r) * b(r) = c(r) + q(r) * g(r)
    let lhs = eval_poly12(&a12, &r) * eval_poly12(&b12, &r);
    let rhs = eval_poly12(&c12, &r) + eval_poly_vec(&q, &r) * g_eval(&r);
    
    assert_eq!(lhs, rhs);
}

#[test]
fn test_schwartz_zippel_rejects_incorrect() {
    let mut rng = test_rng();
    
    // Random a,b in Fq12
    let a = Fq12::rand(&mut rng);
    let b = Fq12::rand(&mut rng);
    let c = a * b;
    
    // Flatten to polynomials
    let a12 = fq12_to_poly12_coeffs(&a);
    let b12 = fq12_to_poly12_coeffs(&b);
    
    // Polynomial multiplication
    let prod = poly_mul(&a12, &b12);
    let g = g_coeffs();
    let (q, _rem) = poly_div_rem_monic(prod, &g);
    
    // Create an incorrect c by adding 1
    let c_bad = c + Fq12::one();
    let c_bad12 = fq12_to_poly12_coeffs(&c_bad);
    
    // Random evaluation point
    let r = Fq::rand(&mut rng);
    
    // Check should fail for incorrect c
    let lhs = eval_poly12(&a12, &r) * eval_poly12(&b12, &r);
    let rhs_bad = eval_poly12(&c_bad12, &r) + eval_poly_vec(&q, &r) * g_eval(&r);
    
    // With overwhelming probability, these should not be equal
    assert_ne!(lhs, rhs_bad);
}

#[test]
fn test_g_polynomial_properties() {
    let mut rng = test_rng();
    let r = Fq::rand(&mut rng);
    
    // Test that g_eval and g_coeffs give the same result
    let g = g_coeffs();
    let g_from_coeffs = eval_poly_vec(&g, &r);
    let g_direct = g_eval(&r);
    
    assert_eq!(g_from_coeffs, g_direct);
    
    // Verify g has the right structure
    assert_eq!(g[0], Fq::from(82u64));
    assert_eq!(g[6], -Fq::from(18u64));
    assert_eq!(g[12], Fq::one());
    
    // All other coefficients should be zero
    for i in 1..6 {
        assert_eq!(g[i], Fq::zero());
    }
    for i in 7..12 {
        assert_eq!(g[i], Fq::zero());
    }
}

/// Comprehensive randomized test that runs multiple trials
fn randomized_trial() -> bool {
    let mut rng = test_rng();
    
    // Random operands and their field product
    let a = Fq12::rand(&mut rng);
    let b = Fq12::rand(&mut rng);
    let c = a * b;
    
    // Flatten into coefficient arrays
    let a12 = fq12_to_poly12_coeffs(&a);
    let b12 = fq12_to_poly12_coeffs(&b);
    let c12 = fq12_to_poly12_coeffs(&c);
    
    // Polynomial multiply in Fq[X]
    let prod = poly_mul(&a12, &b12);
    
    // Divide by g
    let g = g_coeffs();
    let (q, rem) = poly_div_rem_monic(prod, &g);
    
    // Remainder should equal flattened field product c12
    let mut rem_padded = rem.clone();
    rem_padded.resize(12, Fq::zero());
    let rem_ok = rem_padded == c12.to_vec();
    
    // Random-point identity check
    let r = Fq::rand(&mut rng);
    let lhs = eval_poly12(&a12, &r) * eval_poly12(&b12, &r);
    let rhs = eval_poly12(&c12, &r) + eval_poly_vec(&q, &r) * g_eval(&r);
    let point_ok = lhs == rhs;
    
    // "Bad c" should be rejected (adds +1 to c)
    let c_bad = c + Fq12::one();
    let c_bad12 = fq12_to_poly12_coeffs(&c_bad);
    let rhs_bad = eval_poly12(&c_bad12, &r) + eval_poly_vec(&q, &r) * g_eval(&r);
    let rejects_bad = lhs != rhs_bad;
    
    rem_ok && point_ok && rejects_bad
}

#[test]
fn test_many_randomized_trials() {
    // Run many trials to increase confidence
    let trials = 256;
    for i in 0..trials {
        assert!(randomized_trial(), "Failed on trial {}", i);
    }
}

#[test]
fn test_eval_poly12_correctness() {
    let mut rng = test_rng();
    
    // Create a simple polynomial: 1 + 2x + 3x^2
    let mut coeffs = [Fq::zero(); 12];
    coeffs[0] = Fq::from(1u64);
    coeffs[1] = Fq::from(2u64);
    coeffs[2] = Fq::from(3u64);
    
    let r = Fq::from(5u64);
    
    // Manual calculation: 1 + 2*5 + 3*25 = 1 + 10 + 75 = 86
    let expected = Fq::from(86u64);
    let result = eval_poly12(&coeffs, &r);
    
    assert_eq!(result, expected);
}

#[test]
fn test_poly_div_rem_monic_simple() {
    // Test dividing x^3 + 2x^2 + 3x + 4 by x + 1
    // Expected quotient: x^2 + x + 2, remainder: 2
    
    let dividend = vec![
        Fq::from(4u64), // constant
        Fq::from(3u64), // x
        Fq::from(2u64), // x^2
        Fq::from(1u64), // x^3
    ];
    
    let divisor = vec![
        Fq::from(1u64), // constant
        Fq::from(1u64), // x (monic)
    ];
    
    let (q, r) = poly_div_rem_monic(dividend, &divisor);
    
    // Expected quotient: 2 + x + x^2
    assert_eq!(q.len(), 3);
    assert_eq!(q[0], Fq::from(2u64));
    assert_eq!(q[1], Fq::from(1u64));
    assert_eq!(q[2], Fq::from(1u64));
    
    // Expected remainder: 2
    assert_eq!(r.len(), 1);
    assert_eq!(r[0], Fq::from(2u64));
}