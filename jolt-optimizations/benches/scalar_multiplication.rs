use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::{AffineRepr, Group};
use ark_ff::{PrimeField, UniformRand};
use ark_std::test_rng;

use jolt_optimizations::{glv_four_precompute, glv_four_scalar_mul, glv_four_scalar_mul_online};

fn bench_scalar_multiplication(c: &mut Criterion) {
    let mut rng = test_rng();

    // Fix a random scalar for all tests
    let scalar = Fr::rand(&mut rng);

    const NUM_TESTS: usize = 10000;

    // Generate random points
    let points: Vec<G2Projective> = (0..NUM_TESTS)
        .map(|_| G2Affine::rand(&mut rng).into_group())
        .collect();

    let glv_precomputed = glv_four_precompute(&points);

    let mut group = c.benchmark_group("scalar_multiplication");

    group.bench_with_input(
        BenchmarkId::new("naive", NUM_TESTS),
        &points,
        |b, points| {
            b.iter(|| {
                points
                    .par_iter()
                    .map(|point| black_box(point.mul_bigint(scalar.into_bigint())))
                    .collect::<Vec<G2Projective>>()
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("4d_precomputed", NUM_TESTS),
        &glv_precomputed,
        |b, glv_precomputed| {
            b.iter(|| {
                black_box(glv_four_scalar_mul(glv_precomputed, scalar))
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("4d_online", NUM_TESTS),
        &points,
        |b, points| {
            b.iter(|| {
                black_box(glv_four_scalar_mul_online(scalar, points))
            })
        },
    );

    group.finish();
}

criterion_group!(benches, bench_scalar_multiplication);
criterion_main!(benches);
