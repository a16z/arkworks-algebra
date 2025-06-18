use ark_bn254::{Bn254, G1Affine, G2Affine};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::UniformRand;
use ark_std::test_rng;

fn main() {
    println!("Testing BN254 Miller Loop with detailed timing...");
    
    let mut rng = test_rng();
    
    // Create some test points
    let g1_points = [
        G1Affine::rand(&mut rng),
    ];
    let g2_points = [
        G2Affine::rand(&mut rng),
    ];
    
    println!("Calling multi_pairing with {} pairs...", g1_points.len());
    
    // This will trigger our instrumented Miller loop
    let _result = Bn254::multi_pairing(&g1_points, &g2_points);
    
    println!("Test completed!");
}