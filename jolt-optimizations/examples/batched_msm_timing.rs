//! Timing test example for batched MSM
//!
//! This example measures the timing of batched MSM operations with a single run,
//! comparing naive parallel approach vs batched tile-k optimization.

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_std::UniformRand;
use jolt_optimizations::{msm_batched_bn254_tile_k, BatchedMsmConfig};
use rayon::prelude::*;
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    let n = 1 << 11;
    let m = 1 << 11;

    println!("Batched MSM Timing Test");
    println!("Points per MSM (n): {}", n);
    println!("Number of MSMs (m): {}", m);
    println!("{}", "=".repeat(60));

    let mut rng = ark_std::test_rng();

    println!("\nGenerating test data...");
    let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();

    let scalars_owned: Vec<Vec<Fr>> = (0..m)
        .map(|_| (0..n).map(|_| Fr::rand(&mut rng)).collect())
        .collect();

    let scalars_ref: Vec<&[Fr]> = scalars_owned.iter().map(|v| v.as_slice()).collect();

    let cfg = BatchedMsmConfig::default();

    println!("\n1. Naive Parallel MSM:");
    let start = Instant::now();
    let naive_results: Vec<G1Affine> = scalars_owned
        .par_iter()
        .map(|scalars| G1Projective::msm(&bases, scalars).unwrap().into_affine())
        .collect();
    let naive_duration = start.elapsed();
    println!("   Time: {:.2?}", naive_duration);
    println!("   Time per MSM: {:.2?}", naive_duration / m as u32);

    println!("\n2. Batched Tile-K MSM:");
    let start = Instant::now();
    let batched_results = msm_batched_bn254_tile_k(&bases, &scalars_ref, &cfg);
    let batched_duration = start.elapsed();
    println!("   Time: {:.2?}", batched_duration);
    println!("   Time per MSM: {:.2?}", batched_duration / m as u32);

    let speedup = naive_duration.as_secs_f64() / batched_duration.as_secs_f64();
    println!("\n{}", "=".repeat(60));
    println!("Speedup: {:.2}x", speedup);

    assert_eq!(naive_results.len(), batched_results.len());
    for (i, (naive, batched)) in naive_results.iter().zip(batched_results.iter()).enumerate() {
        assert_eq!(naive, batched, "Mismatch at index {}", i);
    }
    println!("Results verified: all MSMs match");
}
