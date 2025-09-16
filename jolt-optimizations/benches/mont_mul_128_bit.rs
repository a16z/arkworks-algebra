use ark_bn254::Fr;
use ark_ff::{BigInt, PrimeField, UniformRand};
use ark_std::rand::RngCore;
use ark_std::test_rng;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_normal_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("mont_mul_128_normal");
    let mut rng = test_rng();

    for size in [100000].iter() {
        let field_elements: Vec<Fr> = (0..*size).map(|_| Fr::rand(&mut rng)).collect();
        let sparse_mont_arrays: Vec<[u64; 4]> = (0..*size)
            .map(|_| [0u64, 0u64, rng.next_u64(), rng.next_u64()])
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                for i in 0..field_elements.len() {
                    let base = field_elements[i];
                    let sparse_mont = sparse_mont_arrays[i];
                    let sparse_bigint = BigInt::<4>(sparse_mont);
                    let sparse_fr = Fr::from_bigint_unchecked(sparse_bigint).unwrap();
                    let _result = black_box(base * sparse_fr);
                }
            });
        });
    }
    group.finish();
}

fn bench_optimized_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("mont_mul_128_optimized");
    let mut rng = test_rng();

    for size in [100000].iter() {
        let field_elements: Vec<Fr> = (0..*size).map(|_| Fr::rand(&mut rng)).collect();
        let sparse_mont_arrays: Vec<[u64; 4]> = (0..*size)
            .map(|_| [0u64, 0u64, rng.next_u64(), rng.next_u64()])
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                for i in 0..field_elements.len() {
                    let base = field_elements[i];
                    let sparse_mont = sparse_mont_arrays[i];
                    let _result = black_box(base.mul_hi_u128(sparse_mont));
                }
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_normal_multiplication,
    bench_optimized_multiplication,
);
criterion_main!(benches);
