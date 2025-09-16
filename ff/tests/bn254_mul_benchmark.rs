#[cfg(test)]
mod tests {
    use ark_test_curves::bn254::Fr;
    use ark_test_curves::bn254::FrConfig;
    use ark_test_curves::MontConfig;

    #[test]
    fn test_bn254_fr_mul_no_carry_opt() {
        #[cfg(feature = "std")]
        {
            use ark_ff::UniformRand;
            use std::time::Instant;

            const NUM_ITERATIONS: usize = 100_000_000;

            let mut rng = ark_std::test_rng();

            let test_pairs: Vec<(Fr, Fr)> = (0..NUM_ITERATIONS)
                .map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)))
                .collect();

            // Test no-carry optimized multiplication
            let mut results_no_carry = Vec::with_capacity(NUM_ITERATIONS);
            let start = Instant::now();
            for (a, b) in &test_pairs {
                let mut result = *a;
                <FrConfig as MontConfig<4>>::mul_assign_no_carry_opt(&mut result, b);
                results_no_carry.push(result);
            }
            let no_carry_time = start.elapsed();

            println!("\n=== BN254 Fr No-Carry Optimized Multiplication ===");
            println!("Iterations: {}", NUM_ITERATIONS);
            println!("Time: {:?}", no_carry_time);
            println!("Per operation: {:?}", no_carry_time / NUM_ITERATIONS as u32);
        }
    }

    #[test]
    fn test_bn254_fr_mul_correctness() {
        // Verify that both multiplication methods produce the same results
        use ark_ff::UniformRand;

        const NUM_TESTS: usize = 10000;

        let mut rng = ark_std::test_rng();

        for _ in 0..NUM_TESTS {
            let a = Fr::rand(&mut rng);
            let b = Fr::rand(&mut rng);

            // Test no-carry optimized multiplication
            let mut result_no_carry = a;
            <FrConfig as MontConfig<4>>::mul_assign_no_carry_opt(&mut result_no_carry, &b);

            // Test standard CIOS multiplication (with optimized Montgomery reduction)
            let mut result_standard = a;
            <FrConfig as MontConfig<4>>::mul_assign_standard_cios(&mut result_standard, &b);

            // Verify results match
            assert_eq!(
                result_no_carry, result_standard,
                "Results differ for a={:?}, b={:?}",
                a, b
            );
        }

        println!(
            "✓ Both multiplication methods produce identical results across {} random tests",
            NUM_TESTS
        );
    }

    #[test]
    fn test_bn254_fr_mul_standard_cios() {
        #[cfg(feature = "std")]
        {
            use ark_ff::UniformRand;
            use std::time::Instant;

            const NUM_ITERATIONS: usize = 100_000_000;

            let mut rng = ark_std::test_rng();

            let test_pairs: Vec<(Fr, Fr)> = (0..NUM_ITERATIONS)
                .map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)))
                .collect();

            // Test standard CIOS multiplication
            let mut results_standard = Vec::with_capacity(NUM_ITERATIONS);
            let start = Instant::now();
            for (a, b) in &test_pairs {
                let mut result = *a;
                <FrConfig as MontConfig<4>>::mul_assign_standard_cios(&mut result, b);
                results_standard.push(result);
            }
            let standard_time = start.elapsed();

            println!("\n=== BN254 Fr Standard CIOS Multiplication ===");
            println!("Iterations: {}", NUM_ITERATIONS);
            println!("Time: {:?}", standard_time);
            println!("Per operation: {:?}", standard_time / NUM_ITERATIONS as u32);
        }
    }
}
