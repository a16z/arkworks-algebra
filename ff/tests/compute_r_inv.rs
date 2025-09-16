#[cfg(test)]
mod tests {
    use ark_ff::{Field, One, PrimeField};
    use ark_test_curves::bn254::Fr;

    #[test]
    fn compute_r_inv_for_bn254_fr() {
        // We need to compute r^{-1} mod p where r = 2^64
        // For BN254 Fr field

        // Create 2^64 as a field element
        let two: Fr = Fr::from(2u64);
        let two_to_64 = {
            let this = &two;
            let exp = &[64u64];
            let mut res = Self::one();

            for i in crate::BitIteratorBE::without_leading_zeros(exp) {
                res.square_in_place();

                if i {
                    res *= this;
                }
            }
            res
        };

        // Compute the inverse
        let r_inv = two_to_64
            .inverse()
            .expect("2^64 should be invertible in Fr");

        // Get the BigInt representation
        let r_inv_bigint = r_inv.into_bigint();

        println!("r^{{-1}} mod p for BN254 Fr:");
        println!("BigInt([");
        for i in 0..4 {
            println!("    0x{:016x},", r_inv_bigint.0[i]);
        }
        println!("])");

        // Also print in decimal for verification
        println!("\nIn decimal:");
        println!("[");
        for i in 0..4 {
            println!("    {},", r_inv_bigint.0[i]);
        }
        println!("]");

        // Verify the computation
        let r_inv_times_r = r_inv * two_to_64;
        assert_eq!(r_inv_times_r, Fr::one(), "r^{{-1}} * r should equal 1");

        println!("\n✓ Verification passed: r^{{-1}} * r = 1 mod p");
    }

    #[test]
    fn compute_r_inv_for_multiple_fields() {
        // We can extend this to compute for other fields if needed
        use ark_test_curves::bls12_381::Fr as BlsFr;

        println!("\n=== BLS12-381 Fr ===");
        let two: BlsFr = BlsFr::from(2u64);
        let two_to_64 = {
            let this = &two;
            let exp = &[64u64];
            let mut res = Self::one();

            for i in crate::BitIteratorBE::without_leading_zeros(exp) {
                res.square_in_place();

                if i {
                    res *= this;
                }
            }
            res
        };
        let r_inv = two_to_64.inverse().expect("2^64 should be invertible");
        let r_inv_bigint = r_inv.into_bigint();

        println!("r^{{-1}} mod p for BLS12-381 Fr:");
        println!("BigInt([");
        for i in 0..4 {
            println!("    0x{:016x},", r_inv_bigint.0[i]);
        }
        println!("])");
    }
}
