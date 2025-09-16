use ark_bn254::Fr;
use ark_ff::{BigInt, PrimeField, UniformRand};
use ark_std::rand::RngCore;
use ark_std::{test_rng, One, Zero};

#[test]
fn test_mont_mul_128_correctness() {
    let mut rng = test_rng();

    // Test with different numbers of random elements
    let test_sizes = vec![10000000];

    for n in test_sizes {
        println!("\n=== Testing with {} random Fr elements ===", n);

        // Generate n random Fr elements
        let field_elements: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        // Generate n sparse Montgomery representations (zeros in low 2 limbs, random in high 2 limbs)
        // These represent field elements that have a specific sparse pattern in Montgomery form
        let sparse_mont_arrays: Vec<[u64; 4]> = (0..n)
            .map(|_| {
                let low_limb = rng.next_u64();
                let high_limb = rng.next_u64();
                [0u64, 0u64, low_limb, high_limb]
            })
            .collect();

        let mut all_tests_passed = true;
        let mut num_tests_passed = 0;

        for i in 0..n {
            let base = field_elements[i];
            let sparse_mont = sparse_mont_arrays[i];

            // Method 1: Normal multiplication
            // Create a full Fr element from the sparse Montgomery representation
            let sparse_bigint = BigInt::<4>(sparse_mont);
            // Use from_bigint_unchecked to interpret this as a Montgomery representation without reduction
            let sparse_fr = Fr::from_bigint_unchecked(sparse_bigint).unwrap();
            let normal_result = base * sparse_fr;

            // Method 2: Optimized multiplication using mul_hi_u128
            // This directly uses the sparse Montgomery representation
            let optimized_result = base.mul_hi_u128(sparse_mont);

            // Assert the results are equal
            if normal_result != optimized_result {
                println!(
                    "Test {} failed!\nBase: {:?}\nSparse Montgomery: {:?}\nNormal: {:?}\nOptimized: {:?}",
                    i, base, sparse_mont, normal_result, optimized_result
                );
                all_tests_passed = false;
            } else {
                num_tests_passed += 1;
            }
        }

        assert!(
            all_tests_passed,
            "Some tests failed for n={}. Passed: {}/{}",
            n, num_tests_passed, n
        );

        println!("✓ All {} tests passed for n={}", n, n);
    }

    println!("\n=== All correctness tests passed! ===");
}

#[test]
fn test_mont_mul_128_edge_cases() {
    let mut rng = test_rng();

    println!("\n=== Testing edge cases ===");

    // Test with zero base
    let zero_base = Fr::zero();
    let sparse = [0u64, 0u64, rng.next_u64(), rng.next_u64()];
    let result = zero_base.mul_hi_u128(sparse);
    assert_eq!(result, Fr::zero(), "Zero base should give zero result");

    // Test with one base
    let one_base = Fr::one();
    let sparse_bigint = BigInt::<4>(sparse);
    let sparse_fr = Fr::from_bigint_unchecked(sparse_bigint).unwrap();
    let result = one_base.mul_hi_u128(sparse);
    assert_eq!(result, sparse_fr, "One base should give sparse value");

    // Test with zero sparse (high limbs are zero)
    let random_base = Fr::rand(&mut rng);
    let zero_sparse = [0u64, 0u64, 0u64, 0u64];
    let result = random_base.mul_hi_u128(zero_sparse);
    assert_eq!(result, Fr::zero(), "Zero sparse should give zero result");

    // Test with max value sparse
    let max_sparse = [0u64, 0u64, u64::MAX, u64::MAX];
    let result = random_base.mul_hi_u128(max_sparse);
    let max_sparse_bigint = BigInt::<4>(max_sparse);
    let max_sparse_fr = Fr::from_bigint_unchecked(max_sparse_bigint).unwrap();
    let expected = random_base * max_sparse_fr;
    assert_eq!(result, expected, "Max sparse value test failed");

    println!("✓ All edge case tests passed");
}
