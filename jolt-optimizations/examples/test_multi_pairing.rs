use ark_bn254::{Bn254, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use ark_std::test_rng;


fn main() {
    println!("Testing BN254 Multi-Pairing...");
    
    let mut rng = test_rng();

     // Create test points
     let g1_affine: Vec<G1Affine> = (0..100000)
     .map(|_| G1Affine::rand(&mut rng))
     .collect();
    let g2_affine: Vec<G2Affine> = (0..100000)
        .map(|_| G2Affine::rand(&mut rng))
        .collect();
    
    // Convert to prepared form
    let g1_prepared: Vec<<Bn254 as Pairing>::G1Prepared> = g1_affine.iter().map(|p| p.into()).collect();
    
    // Use batch conversion for G2 points (parallelized if feature enabled)
    use ark_ec::bn::G2Prepared;
    use ark_bn254::Config;
    let g2_prepared: Vec<<Bn254 as Pairing>::G2Prepared> = G2Prepared::<Config>::batch_from_affine(&g2_affine);
        
    // Test with different numbers of pairs
    for num_pairs in [1] {
        println!("\nTesting with {} pairs:", num_pairs);
        
        // Time the original multi_miller_loop
        let start = std::time::Instant::now();
        let miller_result1 = Bn254::multi_miller_loop(&g1_prepared, &g2_prepared);
        let elapsed_original = start.elapsed();
        
        // Time the optimized multi_miller_loop
        let start = std::time::Instant::now();
        let miller_result2 = Bn254::multi_miller_loop_optimized(&g1_prepared, &g2_prepared);
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