use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::{PrimeField, UniformRand};
use ark_std::{test_rng, Zero};

use jolt_optimizations::{
    fixed_base_vector_msm_g2, DecomposedScalar, FixedBasePrecomputedG2,
};

#[test]
fn test_fixed_base_vector_msm_g2_correctness() {
    let mut rng = test_rng();

    // Generate a fixed base point and multiple scalars
    let base = G2Affine::rand(&mut rng).into_group();
    let scalars: Vec<Fr> = (0..10).map(|_| Fr::rand(&mut rng)).collect();

    // Compute using our optimized fixed-base MSM
    let results_optimized = fixed_base_vector_msm_g2(&base, &scalars);

    // Compute using naive approach for comparison
    let results_naive: Vec<G2Projective> = scalars
        .iter()
        .map(|scalar| base.mul_bigint(scalar.into_bigint()))
        .collect();

    // Verify results match
    for (i, (optimized, naive)) in results_optimized
        .iter()
        .zip(results_naive.iter())
        .enumerate()
    {
        assert_eq!(
            optimized.into_affine(),
            naive.into_affine(),
            "Mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_fixed_base_precomputed_g2() {
    let mut rng = test_rng();
    let base = G2Affine::rand(&mut rng).into_group();

    // Test precomputed interface
    let precomputed = FixedBasePrecomputedG2::new(&base);

    // Test single scalar
    let scalar = Fr::rand(&mut rng);
    let result = precomputed.mul_scalar(scalar);
    let expected = base.mul_bigint(scalar.into_bigint());
    assert_eq!(result.into_affine(), expected.into_affine());

    // Test decomposed scalar
    let decomposed = DecomposedScalar::from_scalar(scalar);
    let result_decomposed = precomputed.mul_scalar_decomposed(&decomposed);
    assert_eq!(result_decomposed.into_affine(), expected.into_affine());

    // Test multiple scalars
    let scalars: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();
    let results = precomputed.mul_scalars(&scalars);
    for (i, (result, scalar)) in results.iter().zip(scalars.iter()).enumerate() {
        let expected = base.mul_bigint(scalar.into_bigint());
        assert_eq!(
            result.into_affine(),
            expected.into_affine(),
            "Mismatch at index {}",
            i
        );
    }

    // Test decomposed scalars
    let decomposed_scalars: Vec<DecomposedScalar> = scalars
        .iter()
        .map(|s| DecomposedScalar::from_scalar(*s))
        .collect();
    let results_decomposed = precomputed.mul_scalars_decomposed(&decomposed_scalars);
    for (i, (result, expected)) in results_decomposed.iter().zip(results.iter()).enumerate() {
        assert_eq!(
            result.into_affine(),
            expected.into_affine(),
            "Decomposed mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_fixed_base_vector_msm_g2_edge_cases() {
    let mut rng = test_rng();
    let base = G2Affine::rand(&mut rng).into_group();

    // Test with single scalar
    let single_scalar = vec![Fr::rand(&mut rng)];
    let single_result = fixed_base_vector_msm_g2(&base, &single_scalar);
    let expected = base.mul_bigint(single_scalar[0].into_bigint());
    assert_eq!(single_result[0].into_affine(), expected.into_affine());

    // Test with zero scalar
    let zero_scalar = vec![Fr::from(0u64)];
    let zero_result = fixed_base_vector_msm_g2(&base, &zero_scalar);
    assert_eq!(zero_result[0], G2Projective::zero());

    // Test with empty vector
    let empty_scalars: Vec<Fr> = vec![];
    let empty_result = fixed_base_vector_msm_g2(&base, &empty_scalars);
    assert!(empty_result.is_empty());
}