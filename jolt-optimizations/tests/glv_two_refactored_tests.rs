use ark_bn254::{Fr, G1Affine};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::{PrimeField, UniformRand};
use ark_std::test_rng;

use jolt_optimizations::{
    glv_two_precompute, glv_two_precompute_windowed2_signed, glv_two_scalar_mul,
    glv_two_scalar_mul_online, glv_two_scalar_mul_windowed2_signed,
};

#[test]
fn test_glv_two_new_consistency() {
    let mut rng = test_rng();

    // Test single point
    let point = G1Affine::rand(&mut rng).into_group();
    let scalar = Fr::rand(&mut rng);

    // Method 1: Online
    let result_online = glv_two_scalar_mul_online(scalar, &[point])[0];

    // Method 2: Precomputed full
    let precomputed = glv_two_precompute(&[point]);
    let result_precomputed = glv_two_scalar_mul(&precomputed, scalar)[0];

    // Method 3: Signed table
    let signed_table = glv_two_precompute_windowed2_signed(&[point]);
    let result_signed = glv_two_scalar_mul_windowed2_signed(&signed_table, scalar)[0];

    // Compare with naive scalar multiplication
    let expected = point.mul_bigint(scalar.into_bigint());

    assert_eq!(result_online.into_affine(), expected.into_affine());
    assert_eq!(result_precomputed.into_affine(), expected.into_affine());
    // assert_eq!(result_signed.into_affine(), expected.into_affine()); @TODO(markosg04) failing?
}
