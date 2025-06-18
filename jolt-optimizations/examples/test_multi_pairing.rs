use ark_bn254::{Bn254, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use ark_std::test_rng;

fn main() {
    println!("Testing BN254 Multi-Pairing...");
    
    let mut rng = test_rng();
    
    // Test with different numbers of pairs
    for num_pairs in [10000] {
        println!("\nTesting with {} pairs:", num_pairs);
        
        // Create test points
        let g1_points: Vec<G1Affine> = (0..num_pairs)
            .map(|_| G1Affine::rand(&mut rng))
            .collect();
        let g2_points: Vec<G2Affine> = (0..num_pairs)
            .map(|_| G2Affine::rand(&mut rng))
            .collect();
        
        // Time the multi_pairing
        let start = std::time::Instant::now();
        let _result = Bn254::multi_pairing(&g1_points, &g2_points);
        let elapsed = start.elapsed();
        
        println!("  Multi-pairing took: {:?}", elapsed);
        println!("  Average per pair: {:?}", elapsed / num_pairs as u32);

        // ~ 100 microseconds per, 200 single threaded
    }
    
    println!("\nTest completed!");
}