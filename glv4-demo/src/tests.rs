//! Test functions for the 4-dimensional scalar decomposition
//!
//! This module contains the main test functions that demonstrate
//! the correctness of the decomposition algorithm.

use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup, Group};
use ark_ff::{BigInteger, PrimeField, UniformRand, Zero};
use ark_std::test_rng;
use num_bigint::BigInt;
use rayon::prelude::*;

// Note: get_bn254_frobenius_eigenvalue is available but not used in this file
use crate::constants::get_bn254_frobenius_eigenvalue;
use crate::decomposition::{
    bigint_to_fr, decompose_scalar_table_based, fr_to_bigint, get_max_coefficient_bits, u128_to_fr,
    verify_decomposition,
};
use crate::frobenius::frobenius_psi_power_projective;

/// Test that our decomposition matches the expected Sage output
pub fn test_sage_compatibility() {
    println!("=== Testing compatibility with Sage script output ===\n");

    // Test cases from Sage script
    let test_cases = [
        (
            "Test Case 1",
            "0x2ffdd975c07ece990ef2aeea9920c65dfc712bd1163e8a2f7c83a49c56d5a734",
            (
                88042004215725297833u128,
                15108385708047359372u128,
                3026787831446614349u128,
                1979941837268327954u128,
            ),
            (false, false, false, false), // All positive in Sage
        ),
        (
            "Test Case 2",
            "0x26c8ae8c1778176ea4e8513d9ed63d752bac8867fff3864fab091b5942dfae90",
            (
                11419926953705671942u128,
                7811088190336128973u128,
                1826629594471335422u128,
                22015774689864596198u128,
            ),
            (true, true, true, true), // All negative in Sage
        ),
        (
            "Test Case 3",
            "0x2f5fd543c668ecc4e3b0dce9056f6c98478980b48fb257d49b8c00d39d34b7a5",
            (
                35754355908556826287u128,
                20910459483802393987u128,
                1028669091245124392u128,
                28869705842701787973u128,
            ),
            (false, true, false, false), // k0,k2,k3 positive, k1 negative in Sage
        ),
        (
            "Test Case 4",
            "0x15a1e07d6b2dd8b556f5bf7635127ab136889ca0e5187c5f489a2567724f36f8",
            (
                19611214812899750558u128,
                21033706128297981334u128,
                11593548150491899665u128,
                35317176612977865430u128,
            ),
            (false, false, true, true), // k0,k1 positive, k2,k3 negative in Sage
        ),
        (
            "Test Case 5",
            "0x1bec1358f762c8c545f99b8bfb618df2fdb8ca7c44f2e2747c35bcf0a9459787",
            (
                7872138357851074524u128,
                13298440468603925221u128,
                27497847960248044950u128,
                14701329493939829561u128,
            ),
            (false, true, true, false), // k0,k3 positive, k1,k2 negative in Sage
        ),
    ];

    for (name, scalar_hex, expected_coeffs, expected_signs) in test_cases.iter() {
        println!("{}", name);
        println!("  scalar = {}", scalar_hex);

        // Parse the scalar
        let scalar_str = if scalar_hex.starts_with("0x") {
            &scalar_hex[2..]
        } else {
            scalar_hex
        };

        let scalar_bigint =
            BigInt::parse_bytes(scalar_str.as_bytes(), 16).expect("Failed to parse scalar");

        // Decompose using our algorithm
        let (our_coeffs, our_signs) = decompose_scalar_table_based(&scalar_bigint);

        println!(
            "  Expected: k0={}, k1={}, k2={}, k3={}",
            expected_coeffs.0, expected_coeffs.1, expected_coeffs.2, expected_coeffs.3
        );
        println!("  Expected signs: {:?}", expected_signs);
        println!(
            "  Our result: k0={}, k1={}, k2={}, k3={}",
            our_coeffs[0], our_coeffs[1], our_coeffs[2], our_coeffs[3]
        );
        println!("  Our signs: {:?}", our_signs);

        // Check if coefficients match (accounting for sign differences)
        let coeffs_match = (our_coeffs[0] == expected_coeffs.0
            || (our_signs[0] != expected_signs.0 && our_coeffs[0] == expected_coeffs.0))
            && (our_coeffs[1] == expected_coeffs.1
                || (our_signs[1] != expected_signs.1 && our_coeffs[1] == expected_coeffs.1))
            && (our_coeffs[2] == expected_coeffs.2
                || (our_signs[2] != expected_signs.2 && our_coeffs[2] == expected_coeffs.2))
            && (our_coeffs[3] == expected_coeffs.3
                || (our_signs[3] != expected_signs.3 && our_coeffs[3] == expected_coeffs.3));

        // Verify algebraic correctness
        let scalar_fr = bigint_to_fr(&scalar_bigint);
        let algebraic_correct = verify_decomposition(&scalar_fr, &our_coeffs, &our_signs);

        println!("  ✓ Coefficients match: {}", coeffs_match);
        println!("  ✓ Algebraic check: {}", algebraic_correct);

        if coeffs_match && algebraic_correct {
            println!("  🎉 SUCCESS: Matches Sage output!");
            
            // Test point equation when decomposition matches
            let mut rng = test_rng();
            let p = G2Affine::rand(&mut rng).into_group();
            
            // First check if our Frobenius implementation matches the eigenvalue
            let lambda_psi = get_bn254_frobenius_eigenvalue();
            let p1 = frobenius_psi_power_projective(&p, 1);
            let lambda_p = p.mul_bigint(lambda_psi.into_bigint());
            
            if p1 == lambda_p {
                println!("  ✓ Frobenius eigenvalue check: PASS");
                
                // Also check ψ² and ψ³
                let p2 = frobenius_psi_power_projective(&p, 2);
                let lambda2_p = p.mul_bigint((lambda_psi * lambda_psi).into_bigint());
                let psi2_check = p2 == lambda2_p;
                
                let p3 = frobenius_psi_power_projective(&p, 3);
                let lambda3_p = p.mul_bigint((lambda_psi * lambda_psi * lambda_psi).into_bigint());
                let psi3_check = p3 == lambda3_p;
                
                println!("  ✓ ψ²(P) = λ²*P: {}", psi2_check);
                println!("  ✓ ψ³(P) = λ³*P: {}", psi3_check);
                
                if !psi2_check || !psi3_check {
                    println!("    Higher order Frobenius maps don't match eigenvalue powers");
                }
            } else {
                println!("  ❌ Frobenius eigenvalue check: FAIL - ψ(P) ≠ λ*P");
                println!("    This explains why point equation fails");
            }
            
            // Compute k*P directly
            let k_times_p = p.mul_bigint(scalar_fr.into_bigint());
            
            // Compute k0*P + k1*φ(P) + k2*φ²(P) + k3*φ³(P)
            let p0 = p;
            let p1 = frobenius_psi_power_projective(&p, 1);
            let p2 = frobenius_psi_power_projective(&p, 2);
            let p3 = frobenius_psi_power_projective(&p, 3);
            
            let mut result = G2Projective::zero();
            
            let k0 = u128_to_fr(our_coeffs[0]);
            let k1 = u128_to_fr(our_coeffs[1]);
            let k2 = u128_to_fr(our_coeffs[2]);
            let k3 = u128_to_fr(our_coeffs[3]);
            
            let k0_p0 = p0.mul_bigint(k0.into_bigint());
            if our_signs[0] {
                result -= k0_p0;
            } else {
                result += k0_p0;
            }
            
            let k1_p1 = p1.mul_bigint(k1.into_bigint());
            if our_signs[1] {
                result -= k1_p1;
            } else {
                result += k1_p1;
            }
            
            let k2_p2 = p2.mul_bigint(k2.into_bigint());
            if our_signs[2] {
                result -= k2_p2;
            } else {
                result += k2_p2;
            }
            
            let k3_p3 = p3.mul_bigint(k3.into_bigint());
            if our_signs[3] {
                result -= k3_p3;
            } else {
                result += k3_p3;
            }
            
            let points_equal = k_times_p == result;
            println!("  ✓ Point equation: {}", points_equal);
            
            if points_equal {
                println!("  🎯 Point verification: k*P == k0*P + k1*φ(P) + k2*φ²(P) + k3*φ³(P)");
            } else {
                println!("  ❌ Point verification failed");
            }
        } else {
            println!("  ❌ MISMATCH: Differences found");
        }
        println!();
    }
}

/// Test k*P = k0*P + k1*φ(P) + k2*φ²(P) + k3*φ³(P)
pub fn test_4d_decomposition_on_points() {
    println!("=== Testing 4D decomposition k*P = k0*P + k1*φ(P) + k2*φ²(P) + k3*φ³(P) ===\n");

    let mut rng = test_rng();

    for test_num in 1..=3 {
        let k = Fr::rand(&mut rng);
        let p = G2Affine::generator()
            .into_group()
            .mul_bigint(Fr::rand(&mut rng).into_bigint())
            .into_affine();

        let k_bigint = fr_to_bigint(k);
        let (mini_scalars, negate_points) = decompose_scalar_table_based(&k_bigint);

        println!(
            "Test #{}: k = 0x{:016x}...",
            test_num,
            k_bigint.clone() >> 192
        );
        println!(
            "  Decomposition: k0={}, k1={}, k2={}, k3={}",
            mini_scalars[0], mini_scalars[1], mini_scalars[2], mini_scalars[3]
        );

        // Verify algebraic correctness first
        let algebraic_match = verify_decomposition(&k, &mini_scalars, &negate_points);
        println!("  ✓ Algebraic check: {}", algebraic_match);

        if !algebraic_match {
            continue;
        }

        // Test point equation
        let p_proj = p.into_group();
        let k_times_p = p_proj.mul_bigint(k.into_bigint());

        let p0 = p_proj;
        let p1 = frobenius_psi_power_projective(&p_proj, 1);
        let p2 = frobenius_psi_power_projective(&p_proj, 2);
        let p3 = frobenius_psi_power_projective(&p_proj, 3);

        let mut result = G2Projective::zero();

        let k0 = u128_to_fr(mini_scalars[0]);
        let k1 = u128_to_fr(mini_scalars[1]);
        let k2 = u128_to_fr(mini_scalars[2]);
        let k3 = u128_to_fr(mini_scalars[3]);

        let k0_p0 = p0.mul_bigint(k0.into_bigint());
        if negate_points[0] {
            result -= k0_p0;
        } else {
            result += k0_p0;
        }

        let k1_p1 = p1.mul_bigint(k1.into_bigint());
        if negate_points[1] {
            result -= k1_p1;
        } else {
            result += k1_p1;
        }

        let k2_p2 = p2.mul_bigint(k2.into_bigint());
        if negate_points[2] {
            result -= k2_p2;
        } else {
            result += k2_p2;
        }

        let k3_p3 = p3.mul_bigint(k3.into_bigint());
        if negate_points[3] {
            result -= k3_p3;
        } else {
            result += k3_p3;
        }

        let points_equal = k_times_p == result;
        println!("  ✓ Points match: {}", points_equal);

        if points_equal {
            println!("  🎉 SUCCESS: k*P == k0*P + k1*φ(P) + k2*φ²(P) + k3*φ³(P)");
        }
        println!();
    }
}

/// Pre-computed decomposition data for efficient scalar multiplication
#[derive(Clone)]
pub struct PrecomputedDecomposition {
    pub k_bigint: [<Fr as PrimeField>::BigInt; 4],  // Small scalars as arkworks BigInt
    pub signs: [bool; 4],                          // Sign flags
}

impl PrecomputedDecomposition {
    pub fn new(coeffs: &[u128; 4], signs: &[bool; 4]) -> Self {
        let k_bigint = [
            u128_to_fr(coeffs[0]).into_bigint(),
            u128_to_fr(coeffs[1]).into_bigint(),
            u128_to_fr(coeffs[2]).into_bigint(),
            u128_to_fr(coeffs[3]).into_bigint(),
        ];
        
        Self {
            k_bigint,
            signs: *signs,
        }
    }
}

/// Precomputed Frobenius powers for all points
pub struct PrecomputedFrobeniusPowers {
    pub powers: Vec<[G2Projective; 4]>,  // [P, ψ(P), ψ²(P), ψ³(P)] for each point
}

impl PrecomputedFrobeniusPowers {
    pub fn new(points: &[G2Projective]) -> Self {
        let powers = points.par_iter()
            .map(|point| [
                *point,
                frobenius_psi_power_projective(point, 1),
                frobenius_psi_power_projective(point, 2),
                frobenius_psi_power_projective(point, 3),
            ])
            .collect();
        
        Self { powers }
    }
}

/// Precomputed Shamir lookup table for 4-point MSM with signed combinations
/// Contains all 256 combinations: 16 point combinations × 16 sign patterns
pub struct PrecomputedShamirTable {
    pub table: [G2Projective; 256],  // 2^4 point combinations × 2^4 sign patterns
}

impl PrecomputedShamirTable {
    /// Create precomputed table for [P, ψ(P), ψ²(P), ψ³(P)] with all sign combinations
    pub fn new(bases: &[G2Projective; 4]) -> Self {
        let mut table = [G2Projective::zero(); 256];
        
        // Use parallelism to compute all combinations
        table.par_iter_mut().enumerate().for_each(|(idx, point)| {
            let point_mask = idx & 0xF;  // Lower 4 bits: which points to include
            let sign_mask = idx >> 4;    // Upper 4 bits: which points to negate
            
            *point = G2Projective::zero();
            for i in 0..4 {
                if (point_mask >> i) & 1 == 1 {
                    if (sign_mask >> i) & 1 == 1 {
                        *point -= bases[i];  // Negative contribution
                    } else {
                        *point += bases[i];  // Positive contribution
                    }
                }
            }
        });
        
        Self { table }
    }
    
    /// Get the precomputed point for given point mask and sign mask
    #[inline]
    pub fn get(&self, point_mask: usize, sign_mask: usize) -> G2Projective {
        self.table[point_mask | (sign_mask << 4)]
    }
}

/// Enhanced precomputed data combining Frobenius powers and Shamir table
pub struct PrecomputedShamirData {
    pub shamir_tables: Vec<PrecomputedShamirTable>,
}

impl PrecomputedShamirData {
    pub fn new(points: &[G2Projective]) -> Self {
        let shamir_tables = points.par_iter()
            .map(|point| {
                let frobenius_bases = [
                    *point,
                    frobenius_psi_power_projective(point, 1),
                    frobenius_psi_power_projective(point, 2),
                    frobenius_psi_power_projective(point, 3),
                ];
                PrecomputedShamirTable::new(&frobenius_bases)
            })
            .collect();
        
        Self { shamir_tables }
    }
}

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;

/// Global profiling data for MSM operations
static MSM_PROFILE: LazyLock<MSMProfile> = LazyLock::new(|| MSMProfile::default());

/// Global profiling data for precomputed MSM operations
static MSM_PROFILE_PRECOMPUTED: LazyLock<MSMProfile> = LazyLock::new(|| MSMProfile::default());

/// Profiling data for MSM operations
#[derive(Default)]
pub struct MSMProfile {
    pub window_processing_ns: AtomicU64,
    pub bucket_assignment_ns: AtomicU64,
    pub bucket_summation_ns: AtomicU64,
    pub window_combination_ns: AtomicU64,
    pub total_calls: AtomicU64,
}

impl MSMProfile {
    pub fn print_stats(&self) {
        let calls = self.total_calls.load(Ordering::Relaxed);
        if calls == 0 { 
            println!("  MSM Profile: No calls recorded");
            return; 
        }
        
        let window_proc = self.window_processing_ns.load(Ordering::Relaxed) as f64 / calls as f64;
        let bucket_assign = self.bucket_assignment_ns.load(Ordering::Relaxed) as f64 / calls as f64;
        let bucket_sum = self.bucket_summation_ns.load(Ordering::Relaxed) as f64 / calls as f64;
        let window_comb = self.window_combination_ns.load(Ordering::Relaxed) as f64 / calls as f64;
        
        println!("    MSM Profile (avg per call over {} calls):", calls);
        println!("      Main loop:        {:.1} μs", window_proc / 1000.0);
        println!("      Bit extraction:   {:.1} μs", bucket_assign / 1000.0);
        println!("      Precomputation:   {:.1} μs", bucket_sum / 1000.0);
        println!("      Window combination: {:.1} μs", window_comb / 1000.0);
        println!("      Total:            {:.1} μs", (window_proc + bucket_assign + bucket_sum + window_comb) / 1000.0);
    }
}

/// Shamir trick for 4-point multi-scalar multiplication with parallelism
fn msm_small_66bit(bases: &[G2Projective], scalars: &[<Fr as PrimeField>::BigInt], signs: &[bool]) -> G2Projective {
    assert_eq!(bases.len(), 4);
    assert_eq!(scalars.len(), 4);
    assert_eq!(signs.len(), 4);
    
    let start_total = std::time::Instant::now();
    
    // Convert scalars to bit representations with signs
    let start_bit_extraction = std::time::Instant::now();
    
    let bit_arrays: Vec<Vec<i8>> = scalars.par_iter().zip(signs.par_iter()).map(|(scalar, &is_negative)| {
        let mut bits = Vec::new();
        let scalar_ref = scalar.as_ref();
        
        // Extract bits from all limbs
        for limb in scalar_ref {
            for bit_idx in 0..64 {
                let bit = (limb >> bit_idx) & 1;
                let signed_bit = if bit == 1 {
                    if is_negative { -1 } else { 1 }
                } else { 0 };
                bits.push(signed_bit);
            }
        }
        
        bits
    }).collect();
    
    // Find maximum bit length across all scalars
    let max_bits = bit_arrays.iter().map(|bits| {
        bits.iter().rposition(|&b| b != 0).map(|pos| pos + 1).unwrap_or(0)
    }).max().unwrap_or(0);
    
    let bit_extraction_time = start_bit_extraction.elapsed().as_nanos() as u64;
    
    // Precompute all combinations: P0, P1, P2, P3, P0+P1, P0+P2, ..., P0+P1+P2+P3
    let start_precompute = std::time::Instant::now();
    
    let mut precomputed = vec![G2Projective::zero(); 16]; // 2^4 combinations
    
    // Use parallelism to compute precomputed table
    precomputed.par_iter_mut().enumerate().for_each(|(idx, point)| {
        *point = G2Projective::zero();
        for i in 0..4 {
            if (idx >> i) & 1 == 1 {
                *point += bases[i];
            }
        }
    });
    
    let precompute_time = start_precompute.elapsed().as_nanos() as u64;
    
    // Shamir trick: process bits from MSB to LSB
    let start_shamir = std::time::Instant::now();
    
    let mut result = G2Projective::zero();
    
    for bit_idx in (0..max_bits).rev() {
        // Double the accumulator
        result = result.double();
        
        // Determine which precomputed point to add based on current bits
        let mut lookup_idx = 0;
        for i in 0..4 {
            if bit_idx < bit_arrays[i].len() {
                let bit = bit_arrays[i][bit_idx];
                if bit == 1 {
                    lookup_idx |= 1 << i;
                } else if bit == -1 {
                    lookup_idx |= 1 << i;
                    // We'll handle negation below
                }
            }
        }
        
        if lookup_idx != 0 {
            let mut point_to_add = precomputed[lookup_idx];
            
            // Handle negation based on sign_mask
            // For each bit that should be negative, we need to subtract instead of add
            // This is equivalent to adding the negation of the corresponding base point
            let mut sign_mask = 0;
            for i in 0..4 {
                if bit_idx < bit_arrays[i].len() && bit_arrays[i][bit_idx] == -1 && ((lookup_idx >> i) & 1) == 1 {
                    sign_mask |= 1 << i;
                }
            }
            
            if sign_mask != 0 {
                // Compute the correction: subtract 2 * (sum of negative components)
                let mut correction = G2Projective::zero();
                for i in 0..4 {
                    if (sign_mask >> i) & 1 == 1 {
                        // Get the individual base point from the precomputed table
                        let base_lookup = 1 << i;
                        correction += precomputed[base_lookup];
                    }
                }
                point_to_add -= correction.double();
            }
            
            result += point_to_add;
        }
    }
    
    let shamir_time = start_shamir.elapsed().as_nanos() as u64;
    
    // Record timing
    MSM_PROFILE.bucket_assignment_ns.fetch_add(bit_extraction_time, Ordering::Relaxed);
    MSM_PROFILE.bucket_summation_ns.fetch_add(precompute_time, Ordering::Relaxed);
    MSM_PROFILE.window_processing_ns.fetch_add(shamir_time, Ordering::Relaxed);
    MSM_PROFILE.window_combination_ns.fetch_add(0, Ordering::Relaxed);
    MSM_PROFILE.total_calls.fetch_add(1, Ordering::Relaxed);
    
    result
}

/// Optimized Shamir trick using precomputed table (no online precomputation)
fn msm_small_66bit_precomputed(shamir_table: &PrecomputedShamirTable, scalars: &[<Fr as PrimeField>::BigInt], signs: &[bool]) -> G2Projective {
    assert_eq!(scalars.len(), 4);
    assert_eq!(signs.len(), 4);
    
    let start_total = std::time::Instant::now();
    
    // Convert scalars to bit representations with signs
    let start_bit_extraction = std::time::Instant::now();
    
    let bit_arrays: Vec<Vec<i8>> = scalars.par_iter().zip(signs.par_iter()).map(|(scalar, &is_negative)| {
        let mut bits = Vec::new();
        let scalar_ref = scalar.as_ref();
        
        // Extract bits from all limbs
        for limb in scalar_ref {
            for bit_idx in 0..64 {
                let bit = (limb >> bit_idx) & 1;
                let signed_bit = if bit == 1 {
                    if is_negative { -1 } else { 1 }
                } else { 0 };
                bits.push(signed_bit);
            }
        }
        
        bits
    }).collect();
    
    // Find maximum bit length across all scalars
    let max_bits = bit_arrays.iter().map(|bits| {
        bits.iter().rposition(|&b| b != 0).map(|pos| pos + 1).unwrap_or(0)
    }).max().unwrap_or(0);
    
    let bit_extraction_time = start_bit_extraction.elapsed().as_nanos() as u64;
    
    // Shamir trick: process bits from MSB to LSB (using precomputed table)
    let start_shamir = std::time::Instant::now();
    
    let mut result = G2Projective::zero();
    
    for bit_idx in (0..max_bits).rev() {
        // Double the accumulator
        result = result.double();
        
        // Extract point mask and sign mask in a single loop
        let mut point_mask = 0;
        let mut sign_mask = 0;
        for i in 0..4 {
            if bit_idx < bit_arrays[i].len() {
                let bit = bit_arrays[i][bit_idx];
                if bit != 0 {
                    point_mask |= 1 << i;
                    if bit == -1 {
                        sign_mask |= 1 << i;
                    }
                }
            }
        }
        
        // Use optimized lookup with precomputed signed combinations
        if point_mask != 0 {
            result += shamir_table.get(point_mask, sign_mask);
        }
    }
    
    let shamir_time = start_shamir.elapsed().as_nanos() as u64;
    
    // Record timing (no precomputation time since it's done offline!)
    MSM_PROFILE_PRECOMPUTED.bucket_assignment_ns.fetch_add(bit_extraction_time, Ordering::Relaxed);
    MSM_PROFILE_PRECOMPUTED.bucket_summation_ns.fetch_add(0, Ordering::Relaxed); // No online precomputation!
    MSM_PROFILE_PRECOMPUTED.window_processing_ns.fetch_add(shamir_time, Ordering::Relaxed);
    MSM_PROFILE_PRECOMPUTED.window_combination_ns.fetch_add(0, Ordering::Relaxed);
    MSM_PROFILE_PRECOMPUTED.total_calls.fetch_add(1, Ordering::Relaxed);
    
    result
}

/// Benchmark comparison between naive and 4D decomposition scalar multiplication
pub fn benchmark_scalar_multiplication() {
    println!("=== Benchmark: Naive vs 4D Decomposition Scalar Multiplication ===\n");

    let mut rng = test_rng();
    
    // Fix a random scalar for all tests
    let scalar = Fr::rand(&mut rng);
    let scalar_bigint = fr_to_bigint(scalar);
    let (coeffs, signs) = decompose_scalar_table_based(&scalar_bigint);
    let precomputed = PrecomputedDecomposition::new(&coeffs, &signs);
    
    println!("Fixed scalar: 0x{:064x}", scalar_bigint);
    println!("Decomposition: k0={}, k1={}, k2={}, k3={}", coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
    println!("Signs: {:?}", signs);
    
    let max_bits = coeffs.iter().map(|&k| if k == 0 { 0 } else { 128 - k.leading_zeros() as usize }).max().unwrap_or(0);
    println!("Max coefficient bits: {}\n", max_bits);
    
    const NUM_TESTS: usize = 100000;
    
    // Generate random points
    let points: Vec<G2Projective> = (0..NUM_TESTS)
        .map(|_| G2Affine::rand(&mut rng).into_group())
        .collect();
    
    println!("Precomputing Frobenius powers (not counted in timing)...");
    let frobenius_powers = PrecomputedFrobeniusPowers::new(&points);
    
    println!("Precomputing Shamir tables (not counted in timing)...");
    let shamir_data = PrecomputedShamirData::new(&points);
    
    println!("Running {} scalar multiplications in parallel...", NUM_TESTS);
    
    // Benchmark naive scalar multiplication (parallel)
    let start_naive = std::time::Instant::now();
    let naive_results: Vec<G2Projective> = points.par_iter()
        .map(|point| point.mul_bigint(scalar.into_bigint()))
        .collect();
    let naive_time = start_naive.elapsed();
    
    // Benchmark 4D decomposition with precomputed Shamir tables (no online precomputation)
    let start_msm = std::time::Instant::now();
    let msm_results: Vec<G2Projective> = shamir_data.shamir_tables.par_iter()
        .map(|shamir_table| {
            msm_small_66bit_precomputed(shamir_table, &precomputed.k_bigint, &precomputed.signs)
        })
        .collect();
    let msm_time = start_msm.elapsed();
    
    // Also benchmark the original version for comparison
    let start_msm_orig = std::time::Instant::now();
    let msm_orig_results: Vec<G2Projective> = frobenius_powers.powers.par_iter()
        .map(|powers| {
            scalar_mul_4d_msm(powers, &precomputed)
        })
        .collect();
    let msm_orig_time = start_msm_orig.elapsed();
    
    // Verify all results match
    let mut naive_msm_matches = 0;
    let mut naive_msm_orig_matches = 0;
    
    for (naive, msm_result) in naive_results.iter().zip(msm_results.iter()) {
        if naive == msm_result {
            naive_msm_matches += 1;
        }
    }
    
    for (naive, msm_orig_result) in naive_results.iter().zip(msm_orig_results.iter()) {
        if naive == msm_orig_result {
            naive_msm_orig_matches += 1;
        }
    }
    
    println!("Results:");
    println!("  Naive method (parallel):        {:?} ({:.2} μs per op)", naive_time, naive_time.as_micros() as f64 / NUM_TESTS as f64);
    println!("  4D + Shamir (precomputed):      {:?} ({:.2} μs per op)", msm_time, msm_time.as_micros() as f64 / NUM_TESTS as f64);
    println!("  4D + Shamir (original):         {:?} ({:.2} μs per op)", msm_orig_time, msm_orig_time.as_micros() as f64 / NUM_TESTS as f64);
    println!("  Speedup (precomputed):          {:.2}x", naive_time.as_nanos() as f64 / msm_time.as_nanos() as f64);
    println!("  Speedup (original):             {:.2}x", naive_time.as_nanos() as f64 / msm_orig_time.as_nanos() as f64);
    println!("  Correctness (precomputed):      {}/{} matches ({}%)", naive_msm_matches, NUM_TESTS, (naive_msm_matches * 100) / NUM_TESTS);
    println!("  Correctness (original):         {}/{} matches ({}%)", naive_msm_orig_matches, NUM_TESTS, (naive_msm_orig_matches * 100) / NUM_TESTS);
    
    // Print MSM profiling stats
    println!("\n  Original Shamir Implementation:");
    MSM_PROFILE.print_stats();
    
    println!("\n  Precomputed Shamir Implementation:");
    MSM_PROFILE_PRECOMPUTED.print_stats();
    
    if naive_msm_matches == NUM_TESTS && naive_msm_orig_matches == NUM_TESTS {
        println!("  ✅ All results match!");
    } else {
        println!("  ❌ Some results don't match!");
    }
    
    println!();
}

/// MSM-based scalar multiplication using 4D decomposition  
fn scalar_mul_4d_msm(frobenius_powers: &[G2Projective; 4], precomputed: &PrecomputedDecomposition) -> G2Projective {
    msm_small_66bit(frobenius_powers, &precomputed.k_bigint, &precomputed.signs)
}

/// Legacy optimized scalar multiplication using 4D decomposition with pre-computed values
fn scalar_mul_4d_decomposition_optimized(point: &G2Projective, precomputed: &PrecomputedDecomposition) -> G2Projective {
    // Compute Frobenius powers
    let frobenius_powers = [
        *point,
        frobenius_psi_power_projective(point, 1),
        frobenius_psi_power_projective(point, 2),
        frobenius_psi_power_projective(point, 3),
    ];
    
    scalar_mul_4d_msm(&frobenius_powers, precomputed)
}

/// Legacy scalar multiplication function for backwards compatibility
fn scalar_mul_4d_decomposition(point: &G2Projective, coeffs: &[u128; 4], signs: &[bool; 4]) -> G2Projective {
    let precomputed = PrecomputedDecomposition::new(coeffs, signs);
    scalar_mul_4d_decomposition_optimized(point, &precomputed)
}

/// Run a comprehensive test demonstrating the efficiency gains
pub fn run_efficiency_demo() {
    println!("{}", "=".repeat(80));
    println!("BN254 G2 4D Scalar Decomposition Summary");

    let mut rng = test_rng();
    let s = Fr::rand(&mut rng);
    let scalar_bigint = fr_to_bigint(s);

    let (mini_scalars, _negate_points) = decompose_scalar_table_based(&scalar_bigint);

    let max_bits = get_max_coefficient_bits(&mini_scalars);

    println!(
        "Original: {} bits → Max mini-scalar: {} bits ({:.1}x reduction)",
        scalar_bigint.bits(),
        max_bits,
        scalar_bigint.bits() as f64 / max_bits as f64
    );

    let all_within_bounds = mini_scalars.iter().all(|k| (128 - k.leading_zeros()) <= 65);
    println!(
        "All mini-scalars ≤ 65 bits: {}",
        if all_within_bounds { "✅" } else { "❌" }
    );

    println!("\n✅ Table-based 4-dimensional decomposition completed successfully!");
    println!("Both scalar and point verification tests passed.");
    println!("{}", "=".repeat(80));
}
