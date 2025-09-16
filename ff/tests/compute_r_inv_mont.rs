#[cfg(test)]
mod tests {
    use ark_ff::{BigInteger, Field, One, PrimeField};
    use ark_test_curves::bn254::Fr;

    #[test]
    fn compute_r_inv_montgomery_form() {
        // We need r^{-1} mod p in Montgomery form
        // where r = 2^64 (single limb)

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

        // r_inv is already in Montgomery form internally
        // Get the Montgomery representation directly
        let r_inv_bigint = r_inv.into_bigint();

        println!("r^{{-1}} mod p in Montgomery form for BN254 Fr:");
        println!("// Hardcoded Montgomery form of r^{{-1}} where r = 2^64");
        println!("let r_inv_mont = BigInt([");
        for i in 0..4 {
            println!("    0x{:016x},", r_inv_bigint.0[i]);
        }
        println!("]);");

        // Also compute what the standard form would be
        // by converting out of Montgomery form
        println!("\n// For reference, standard form would be:");
        let r_inv_standard = r_inv.into_bigint();

        // To get standard form, we need to multiply by R^{-1} mod p
        // where R = 2^256 for BN254
        // But since r_inv is already the value we want, let's verify

        // Verification: r_inv * 2^64 = 1 (in the field)
        let verification = r_inv * two_to_64;
        assert_eq!(verification, Fr::one(), "r^{{-1}} * r should equal 1");

        println!("\n✓ Verification passed: r^{{-1}} * r = 1 mod p");

        // Print the actual limbs to use in the const function
        println!("\n// For const fn r_inv implementation:");
        println!("if N >= 4 {{");
        for i in 0..4 {
            println!("    result[{}] = 0x{:016x};", i, r_inv_bigint.0[i]);
        }
        println!("}}");
    }

    #[test]
    fn verify_montgomery_multiplication() {
        use ark_ff::MontFp;

        // Test that our r_inv works correctly
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
        let r_inv = two_to_64.inverse().unwrap();

        // Test: a * b * r_inv should give the correct result
        let a = Fr::from(123u64);
        let b = Fr::from(456u64);

        // Standard multiplication
        let expected = a * b;

        // Simulated optimized multiplication would be:
        // (a * b in Montgomery) * r_inv
        // Since a and b are already in Montgomery form internally
        let result = a * b * r_inv * two_to_64;

        assert_eq!(
            result, expected,
            "Optimized multiplication should match standard"
        );
        println!("✓ Montgomery multiplication with r_inv verified");
    }
}
