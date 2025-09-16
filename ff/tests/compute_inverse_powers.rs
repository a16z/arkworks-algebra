#[cfg(test)]
mod tests {
    use ark_ff::{Field, One, PrimeField};
    use ark_test_curves::bn254::Fr;

    #[test]
    fn compute_inverse_powers_for_bn254() {
        // Compute r^{-1}, r^{-2}, r^{-3} mod p where r = 2^64

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

        // r^{-1} mod p
        let r_inv = two_to_64.inverse().expect("2^64 should be invertible");
        let r_inv_bigint = r_inv.into_bigint();

        println!("I1 (r^{{-1}} mod p) in Montgomery form:");
        println!("const I1: [u64; 4] = [");
        for i in 0..4 {
            println!("    0x{:016x},", r_inv_bigint.0[i]);
        }
        println!("];");

        // r^{-2} mod p
        let r_inv_2 = r_inv * r_inv;
        let r_inv_2_bigint = r_inv_2.into_bigint();

        println!("\nI2 (r^{{-2}} mod p) in Montgomery form:");
        println!("const I2: [u64; 4] = [");
        for i in 0..4 {
            println!("    0x{:016x},", r_inv_2_bigint.0[i]);
        }
        println!("];");

        // r^{-3} mod p
        let r_inv_3 = r_inv_2 * r_inv;
        let r_inv_3_bigint = r_inv_3.into_bigint();

        println!("\nI3 (r^{{-3}} mod p) in Montgomery form:");
        println!("const I3: [u64; 4] = [");
        for i in 0..4 {
            println!("    0x{:016x},", r_inv_3_bigint.0[i]);
        }
        println!("];");

        // Verify
        let one = Fr::one();
        assert_eq!(r_inv * two_to_64, one);
        assert_eq!(r_inv_2 * two_to_64 * two_to_64, one);
        assert_eq!(r_inv_3 * two_to_64 * two_to_64 * two_to_64, one);

        println!("\n✓ All verifications passed");
    }
}
