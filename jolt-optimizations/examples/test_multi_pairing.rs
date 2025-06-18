use ark_bn254::{Bn254, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use ark_std::test_rng;

fn main() {
    println!("Testing BN254 Multi-Pairing...");
    
    let mut rng = test_rng();
    
    // Test with different numbers of pairs
    for num_pairs in [100] {
        println!("\nTesting with {} pairs:", num_pairs);
        
        // Create test points
        let g1_points: Vec<G1Affine> = (0..num_pairs)
            .map(|_| G1Affine::rand(&mut rng))
            .collect();
        let g2_points: Vec<G2Affine> = (0..num_pairs)
            .map(|_| G2Affine::rand(&mut rng))
            .collect();
        
        // Time the original multi_miller_loop
        let start = std::time::Instant::now();
        let miller_result1 = Bn254::multi_miller_loop(&g1_points, &g2_points);
        let elapsed_original = start.elapsed();
        
        // Time the optimized multi_miller_loop
        let start = std::time::Instant::now();
        let miller_result2 = Bn254::multi_miller_loop_optimized(&g1_points, &g2_points);
        let elapsed_optimized = start.elapsed();
        
        // Apply final exponentiation to both results
        let result1 = Bn254::final_exponentiation(miller_result1).unwrap();
        let result2 = Bn254::final_exponentiation(miller_result2).unwrap();
        
        // Verify results match
        assert_eq!(result1, result2, "Results don't match for {} pairs!", num_pairs);
        
        println!("  Original multi_miller_loop: {:?}", elapsed_original);
        println!("  Optimized multi_miller_loop: {:?}", elapsed_optimized);
        println!("  Speedup: {:.2}x", elapsed_original.as_secs_f64() / elapsed_optimized.as_secs_f64());
        println!("  Average per pair (original): {:?}", elapsed_original / num_pairs as u32);
        println!("  Average per pair (optimized): {:?}", elapsed_optimized / num_pairs as u32);
    }
    
    println!("\nTest completed! All results match.");
}