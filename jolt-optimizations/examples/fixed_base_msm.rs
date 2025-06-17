//! Example demonstrating the fixed-base MSM interface with separated precomputation
//! This shows how libraries can precompute on group elements and later apply scalars

use ark_bn254::{Fr, G1Affine, G2Affine};
use ark_ec::AffineRepr;
use ark_ff::UniformRand;
use ark_std::test_rng;
use jolt_optimizations::{
    DecomposedScalar, DecomposedScalar2D, FixedBasePrecomputedG1, FixedBasePrecomputedG2,
};

fn main() {
    let mut rng = test_rng();
    
    println!("=== Fixed-Base MSM Example ===\n");
    
    // Example 1: G2 Fixed-Base MSM
    println!("G2 Example:");
    let g2_base = G2Affine::rand(&mut rng).into_group();
    
    // Step 1: Precompute on the group element (done once)
    println!("  1. Precomputing Frobenius endomorphisms for G2 base...");
    let g2_precomputed = FixedBasePrecomputedG2::new(&g2_base);
    
    // Step 2: Later, when scalars are known, decompose them
    println!("  2. Decomposing scalars...");
    let scalars: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();
    let decomposed_scalars: Vec<DecomposedScalar> = scalars
        .iter()
        .map(|s| DecomposedScalar::from_scalar(*s))
        .collect();
    
    // Step 3: Apply the decomposed scalars efficiently
    println!("  3. Computing scalar multiplications...");
    let results = g2_precomputed.mul_scalars_decomposed(&decomposed_scalars);
    
    println!("  ✓ Computed {} G2 scalar multiplications\n", results.len());
    
    // Example 2: G1 Fixed-Base MSM
    println!("G1 Example:");
    let g1_base = G1Affine::rand(&mut rng).into_group();
    
    // Step 1: Precompute on the group element
    println!("  1. Precomputing GLV endomorphisms for G1 base...");
    let g1_precomputed = FixedBasePrecomputedG1::new(&g1_base);
    
    // Step 2: Decompose scalars when available
    println!("  2. Decomposing scalars...");
    let g1_decomposed_scalars: Vec<DecomposedScalar2D> = scalars
        .iter()
        .map(|s| DecomposedScalar2D::from_scalar(*s))
        .collect();
    
    // Step 3: Apply the decomposed scalars
    println!("  3. Computing scalar multiplications...");
    let g1_results = g1_precomputed.mul_scalars_decomposed(&g1_decomposed_scalars);
    
    println!("  ✓ Computed {} G1 scalar multiplications\n", g1_results.len());
    
    // Example 3: Single scalar multiplication
    println!("Single Scalar Example:");
    let single_scalar = Fr::rand(&mut rng);
    
    // Using precomputed data for a single scalar
    let g2_single_result = g2_precomputed.mul_scalar(single_scalar);
    let g1_single_result = g1_precomputed.mul_scalar(single_scalar);
    
    println!("  ✓ Computed single G2 scalar multiplication");
    println!("  ✓ Computed single G1 scalar multiplication\n");
    
    // Example 4: Reusing precomputed data with new scalars
    println!("Reusing Precomputed Data:");
    let new_scalars: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();
    
    // The same precomputed data can be reused with different scalars
    let new_g2_results = g2_precomputed.mul_scalars(&new_scalars);
    let new_g1_results = g1_precomputed.mul_scalars(&new_scalars);
    
    println!("  ✓ Reused G2 precomputed data for {} new scalars", new_g2_results.len());
    println!("  ✓ Reused G1 precomputed data for {} new scalars", new_g1_results.len());
    
    println!("\n=== Summary ===");
    println!("The new interface allows:");
    println!("1. One-time precomputation on group elements");
    println!("2. Separate scalar decomposition when scalars are known");
    println!("3. Efficient application of decomposed scalars");
    println!("4. Reuse of precomputed data with different scalars");
}