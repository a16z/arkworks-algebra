//! Dory-specific optimized utilities for vector scalar multiplication
//!
//! This module provides specialized interfaces for common Dory operations,
//! particularly vector scalar multiplication with fixed scalars and generators.

use ark_bn254::{Fr, G2Projective};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
// use ark_ec::Group;
use ark_std::Zero;
use ark_ec::PrimeGroup;

use crate::decomposition::{decompose_scalar_table_based, fr_to_bigint, u128_to_fr};
use crate::frobenius::frobenius_psi_power_projective;
use crate::glv_four::{shamir_glv_mul_precomputed, PrecomputedShamirData, PrecomputedShamirTable};

/// Helper function to decompose a scalar into 4D GLV form
fn decompose_scalar(scalar: Fr) -> ([<Fr as PrimeField>::BigInt; 4], [bool; 4]) {
    let scalar_bigint = fr_to_bigint(scalar);
    let (coeffs, signs) = decompose_scalar_table_based(&scalar_bigint);

    let k_bigint = [
        u128_to_fr(coeffs[0]).into_bigint(),
        u128_to_fr(coeffs[1]).into_bigint(),
        u128_to_fr(coeffs[2]).into_bigint(),
        u128_to_fr(coeffs[3]).into_bigint(),
    ];

    (k_bigint, signs)
}

/// Precomputed data for efficient vector scalar multiplication with a fixed scalar
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct VectorScalarMulData {
    /// Decomposed scalar coefficients
    pub scalar_coeffs: [<Fr as PrimeField>::BigInt; 4],
    /// Signs for each coefficient
    pub scalar_signs: [bool; 4],
    /// Precomputed Shamir tables for each generator
    pub precomputed_data: PrecomputedShamirData,
}

impl VectorScalarMulData {
    /// Create precomputed data for vector scalar multiplication
    ///
    /// # Arguments
    /// * `generators` - Fixed G2 generators that will be multiplied
    /// * `scalar` - Fixed scalar that will be used for all multiplications
    pub fn new(generators: &[G2Projective], scalar: Fr) -> Self {
        // Decompose the scalar once
        let (scalar_coeffs, scalar_signs) = decompose_scalar(scalar);

        // Precompute Shamir tables for all generators
        let precomputed_data = PrecomputedShamirData::new(generators);

        Self {
            scalar_coeffs,
            scalar_signs,
            precomputed_data,
        }
    }

    /// Get the number of generators this data was created for
    pub fn num_generators(&self) -> usize {
        self.precomputed_data.shamir_tables.len()
    }
}

/// Perform vector scalar multiplication and addition using precomputed data
///
/// Computes `v[i] = v[i] + scalar * generators[i]` for all i, where scalar and generators
/// are fixed and precomputed in `data`.
///
/// # Arguments
/// * `v` - Mutable reference to vector to update (values will be added to)
/// * `data` - Precomputed data containing decomposed scalar and generator tables
///
/// # Panics
/// * If `v.len() != data.num_generators()`
pub fn vector_scalar_mul_add_precomputed(v: &mut [G2Projective], data: &VectorScalarMulData) {
    assert_eq!(
        v.len(),
        data.num_generators(),
        "Vector length must match number of precomputed generators"
    );

    use rayon::prelude::*;

    // Perform scalar multiplication and addition in parallel
    v.par_iter_mut().enumerate().for_each(|(i, v_point)| {
        let scalar_mul_result = shamir_glv_mul_precomputed(
            &data.precomputed_data.shamir_tables[i],
            &data.scalar_coeffs,
            &data.scalar_signs,
        );
        *v_point += scalar_mul_result;
    });
}

/// Perform vector scalar multiplication and addition online (without precomputation)
///
/// Computes `v[i] = v[i] + scalar * generators[i]` for all i.
/// This version decomposes the scalar once but doesn't use precomputed tables.
///
/// # Arguments
/// * `v` - Mutable reference to vector to update (values will be added to)
/// * `generators` - Fixed G2 generators to multiply
/// * `scalar` - Fixed scalar to multiply with each generator
///
/// # Panics
/// * If `v.len() != generators.len()`
pub fn vector_scalar_mul_add_online(
    v: &mut [G2Projective],
    generators: &[G2Projective],
    scalar: Fr,
) {
    assert_eq!(
        v.len(),
        generators.len(),
        "Vector and generators must have the same length"
    );

    use rayon::prelude::*;

    // Decompose the scalar once
    let (scalar_coeffs, scalar_signs) = decompose_scalar(scalar);

    // Perform scalar multiplication and addition in parallel
    v.par_iter_mut()
        .zip(generators.par_iter())
        .for_each(|(v_point, generator)| {
            // Compute Frobenius bases for this generator
            let frobenius_bases = [
                *generator,
                frobenius_psi_power_projective(generator, 1),
                frobenius_psi_power_projective(generator, 2),
                frobenius_psi_power_projective(generator, 3),
            ];

            // Create temporary Shamir table
            let shamir_table = PrecomputedShamirTable::new(&frobenius_bases);

            // Perform scalar multiplication and add to existing value
            let scalar_mul_result =
                shamir_glv_mul_precomputed(&shamir_table, &scalar_coeffs, &scalar_signs);
            *v_point += scalar_mul_result;
        });
}

/// Convenience function to create and use precomputed data in one call
///
/// This is equivalent to creating `VectorScalarMulData` and then calling
/// `vector_scalar_mul_add_precomputed`, but more convenient for one-time use.
///
/// # Arguments
/// * `v` - Mutable reference to vector to update (values will be added to)
/// * `generators` - Fixed G2 generators to multiply
/// * `scalar` - Fixed scalar to multiply with each generator
pub fn vector_scalar_mul_add(v: &mut [G2Projective], generators: &[G2Projective], scalar: Fr) {
    let data = VectorScalarMulData::new(generators, scalar);
    vector_scalar_mul_add_precomputed(v, &data);
}

/// Precomputed data for efficient vector scalar multiplication where we scale the vector elements
/// and add generators: v[i] = scalar * v[i] + generators[i]
#[derive(Clone, Debug)]
pub struct VectorScalarMulVData {
    /// Decomposed scalar coefficients
    pub scalar_coeffs: [<Fr as PrimeField>::BigInt; 4],
    /// Signs for each coefficient
    pub scalar_signs: [bool; 4],
}

impl VectorScalarMulVData {
    /// Create precomputed scalar decomposition for vector element scaling
    ///
    /// # Arguments
    /// * `scalar` - Fixed scalar that will be used to scale vector elements
    pub fn new(scalar: Fr) -> Self {
        let (scalar_coeffs, scalar_signs) = decompose_scalar(scalar);
        
        Self {
            scalar_coeffs,
            scalar_signs,
        }
    }
}

/// Perform vector scalar multiplication with vector scaling using precomputed data
///
/// Computes `v[i] = scalar * v[i] + generators[i]` for all i, where scalar decomposition
/// is precomputed in `data`.
///
/// # Arguments
/// * `v` - Mutable reference to vector to update (will be scaled and then added to)
/// * `generators` - Fixed G2 generators to add to scaled vector elements
/// * `data` - Precomputed data containing decomposed scalar
///
/// # Panics
/// * If `v.len() != generators.len()`
pub fn vector_scalar_mul_v_add_g_precomputed(
    v: &mut [G2Projective],
    generators: &[G2Projective],
    data: &VectorScalarMulVData,
) {
    assert_eq!(
        v.len(),
        generators.len(),
        "Vector and generators must have the same length"
    );

    use rayon::prelude::*;

    // Perform scalar multiplication and addition in parallel
    v.par_iter_mut()
        .zip(generators.par_iter())
        .for_each(|(v_point, generator)| {
            // Compute Frobenius bases for current vector element
            let frobenius_bases = [
                *v_point,
                frobenius_psi_power_projective(v_point, 1),
                frobenius_psi_power_projective(v_point, 2),
                frobenius_psi_power_projective(v_point, 3),
            ];

            // Create temporary Shamir table for v_point
            let shamir_table = PrecomputedShamirTable::new(&frobenius_bases);

            // Perform scalar multiplication: scalar * v[i] + generators[i]
            let v_scaled = shamir_glv_mul_precomputed(&shamir_table, &data.scalar_coeffs, &data.scalar_signs);
            *v_point = v_scaled + generator;
        });
}

/// Perform vector scalar multiplication with vector scaling online (without precomputation)
///
/// Computes `v[i] = scalar * v[i] + generators[i]` for all i.
/// This version decomposes the scalar once but doesn't use precomputed tables.
///
/// # Arguments
/// * `v` - Mutable reference to vector to update (will be scaled and then added to)
/// * `generators` - Fixed G2 generators to add to scaled vector elements  
/// * `scalar` - Fixed scalar to multiply with each vector element
///
/// # Panics
/// * If `v.len() != generators.len()`
pub fn vector_scalar_mul_v_add_g_online(
    v: &mut [G2Projective],
    generators: &[G2Projective],
    scalar: Fr,
) {
    assert_eq!(
        v.len(),
        generators.len(),
        "Vector and generators must have the same length"
    );

    let data = VectorScalarMulVData::new(scalar);
    vector_scalar_mul_v_add_g_precomputed(v, generators, &data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::G2Affine;
    use ark_ec::{AdditiveGroup, AffineRepr, CurveGroup};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_vector_scalar_mul_add_consistency() {
        let mut rng = test_rng();

        // Generate test data
        let num_generators = 10;
        let generators: Vec<G2Projective> = (0..num_generators)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();
        let scalar = Fr::rand(&mut rng);

        // Generate initial values
        let initial_values: Vec<G2Projective> = (0..num_generators)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();

        // Test online version
        let mut v_online = initial_values.clone();
        vector_scalar_mul_add_online(&mut v_online, &generators, scalar);

        // Test precomputed version
        let mut v_precomputed = initial_values.clone();
        vector_scalar_mul_add(&mut v_precomputed, &generators, scalar);

        // Test with separate data creation
        let data = VectorScalarMulData::new(&generators, scalar);
        let mut v_separate = initial_values.clone();
        vector_scalar_mul_add_precomputed(&mut v_separate, &data);

        // Compare with naive computation: v[i] + scalar * generators[i]
        for i in 0..num_generators {
            let expected = initial_values[i] + generators[i].mul_bigint(scalar.into_bigint());

            // Convert to affine for comparison
            let expected_affine = expected.into_affine();
            let online_affine = v_online[i].into_affine();
            let precomputed_affine = v_precomputed[i].into_affine();
            let separate_affine = v_separate[i].into_affine();

            assert_eq!(
                expected_affine, online_affine,
                "Online version mismatch at index {}",
                i
            );
            assert_eq!(
                expected_affine, precomputed_affine,
                "Precomputed version mismatch at index {}",
                i
            );
            assert_eq!(
                expected_affine, separate_affine,
                "Separate data version mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_edge_cases() {
        let mut rng = test_rng();

        // Test with single generator
        let generators = vec![G2Affine::rand(&mut rng).into_group()];
        let scalar = Fr::rand(&mut rng);
        let initial = G2Affine::rand(&mut rng).into_group();

        let mut v = vec![initial];
        vector_scalar_mul_add_online(&mut v, &generators, scalar);

        let expected = initial + generators[0].mul_bigint(scalar.into_bigint());
        assert_eq!(v[0].into_affine(), expected.into_affine());

        // Test with zero scalar (should just keep initial value)
        let scalar_zero = Fr::from(0u64);
        let mut v_zero = vec![initial];
        vector_scalar_mul_add_online(&mut v_zero, &generators, scalar_zero);
        assert_eq!(v_zero[0].into_affine(), initial.into_affine());

        // Test with identity generator (should keep initial value)
        let identity_generators = vec![G2Projective::zero()];
        let mut v_identity = vec![initial];
        vector_scalar_mul_add_online(&mut v_identity, &identity_generators, scalar);
        assert_eq!(v_identity[0].into_affine(), initial.into_affine());

        // Test with zero initial value
        let mut v_zero_init = vec![G2Projective::zero()];
        vector_scalar_mul_add_online(&mut v_zero_init, &generators, scalar);
        let expected_zero_init = generators[0].mul_bigint(scalar.into_bigint());
        assert_eq!(
            v_zero_init[0].into_affine(),
            expected_zero_init.into_affine()
        );
    }

    #[test]
    fn test_vector_scalar_mul_v_add_g_consistency() {
        let mut rng = test_rng();

        // Generate test data
        let num_generators = 10;
        let generators: Vec<G2Projective> = (0..num_generators)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();
        let scalar = Fr::rand(&mut rng);

        // Generate initial values
        let initial_values: Vec<G2Projective> = (0..num_generators)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();

        // Test online version
        let mut v_online = initial_values.clone();
        vector_scalar_mul_v_add_g_online(&mut v_online, &generators, scalar);

        // Test precomputed version
        let mut v_precomputed = initial_values.clone();
        let data = VectorScalarMulVData::new(scalar);
        vector_scalar_mul_v_add_g_precomputed(&mut v_precomputed, &generators, &data);

        // Compare with naive computation: scalar * v[i] + generators[i]
        for i in 0..num_generators {
            let expected = initial_values[i].mul_bigint(scalar.into_bigint()) + generators[i];

            // Convert to affine for comparison
            let expected_affine = expected.into_affine();
            let online_affine = v_online[i].into_affine();
            let precomputed_affine = v_precomputed[i].into_affine();

            assert_eq!(
                expected_affine, online_affine,
                "Online version mismatch at index {}",
                i
            );
            assert_eq!(
                expected_affine, precomputed_affine,
                "Precomputed version mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_vector_scalar_mul_v_add_g_edge_cases() {
        let mut rng = test_rng();

        // Test with single generator
        let generators = vec![G2Affine::rand(&mut rng).into_group()];
        let scalar = Fr::rand(&mut rng);
        let initial = G2Affine::rand(&mut rng).into_group();

        let mut v = vec![initial];
        vector_scalar_mul_v_add_g_online(&mut v, &generators, scalar);

        let expected = initial.mul_bigint(scalar.into_bigint()) + generators[0];
        assert_eq!(v[0].into_affine(), expected.into_affine());

        // Test with zero scalar (should just be generators[i])
        let scalar_zero = Fr::from(0u64);
        let mut v_zero = vec![initial];
        vector_scalar_mul_v_add_g_online(&mut v_zero, &generators, scalar_zero);
        assert_eq!(v_zero[0].into_affine(), generators[0].into_affine());

        // Test with identity generator (should just be scalar * v[i])
        let identity_generators = vec![G2Projective::zero()];
        let mut v_identity = vec![initial];
        vector_scalar_mul_v_add_g_online(&mut v_identity, &identity_generators, scalar);
        let expected_identity = initial.mul_bigint(scalar.into_bigint());
        assert_eq!(
            v_identity[0].into_affine(),
            expected_identity.into_affine()
        );

        // Test with zero initial value (should just be generators[i])
        let mut v_zero_init = vec![G2Projective::zero()];
        vector_scalar_mul_v_add_g_online(&mut v_zero_init, &generators, scalar);
        assert_eq!(
            v_zero_init[0].into_affine(),
            generators[0].into_affine()
        );
    }
}
