//! 4-dimensional scalar decomposition for BN254 G2 using precomputed lookup table
//!
//! This implements the decomposition k*P = k0*P + k1*φ(P) + k2*φ²(P) + k3*φ³(P)
//! where φ is the Frobenius endomorphism, using precomputed decompositions for powers of 2

mod constants;
mod decomposition;
mod frobenius;
mod tests;

use ark_bn254::Fr;
use ark_ff::PrimeField;

use crate::constants::get_bn254_frobenius_eigenvalue;
use crate::decomposition::fr_to_bigint;
use crate::tests::{benchmark_scalar_multiplication, run_efficiency_demo, test_4d_decomposition_on_points, test_sage_compatibility};

fn main() {
    println!("BN254 G2 4-dimensional scalar decomposition using precomputed lookup table");
    println!("{}\n", "=".repeat(80));

    let lambda_psi = get_bn254_frobenius_eigenvalue();
    println!(
        "ψ endomorphism eigenvalue λ_ψ: 0x{:064x}",
        fr_to_bigint(lambda_psi)
    );

    // Verify λ_ψ^4 mod r
    let lambda_psi_4 = lambda_psi * lambda_psi * lambda_psi * lambda_psi;
    let expected_lambda4 = Fr::from_be_bytes_mod_order(&[
        0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29, 0x04, 0x8b, 0x6e, 0x19, 0x3f, 0xd8, 0x41,
        0x04, 0xcc, 0x37, 0xa7, 0x3f, 0xec, 0x2b, 0xc5, 0xe9, 0xb8, 0xca, 0x0b, 0x2d, 0x36, 0x63,
        0x6f, 0x23,
    ]);
    println!(
        "λ_ψ^4 check: {}\n",
        if lambda_psi_4 == expected_lambda4 {
            "✓"
        } else {
            "✗"
        }
    );

    // Test compatibility with Sage script
    test_sage_compatibility();

    // Test the point equation
    test_4d_decomposition_on_points();

    // Run benchmark comparison
    benchmark_scalar_multiplication();

    // Run efficiency demonstration
    run_efficiency_demo();
}
