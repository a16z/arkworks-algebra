use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Instant;

use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::{AffineRepr, AdditiveGroup};
use ark_ff::{PrimeField, UniformRand};
use ark_std::test_rng;
use ark_ec::PrimeGroup;

use jolt_optimizations::{
    vector_scalar_mul_add, vector_scalar_mul_add_online, vector_scalar_mul_add_precomputed,
    vector_scalar_mul_v_add_g_online, vector_scalar_mul_v_add_g_precomputed,
    VectorScalarMulData, VectorScalarMulVData,
};

fn bench_vector_scalar_mul_add(c: &mut Criterion) {
    let mut rng = test_rng();

    // Test with different vector sizes
    let vector_sizes = [1000];

    for &size in &vector_sizes {
        // Generate test data
        let generators: Vec<G2Projective> = (0..size)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();
        let scalar = Fr::rand(&mut rng);

        // Initial values for the vector
        let initial_values: Vec<G2Projective> = (0..size)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();

        let mut group = c.benchmark_group(format!("vector_scalar_mul_add_{}", size));

        // Benchmark naive approach
        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &(&generators, &scalar, &initial_values),
            |b, &(generators, scalar, initial_values)| {
                b.iter(|| {
                    let mut v = initial_values.clone();
                    // Naive implementation: v[i] += scalar * generators[i]
                    for (v_i, g_i) in v.iter_mut().zip(generators.iter()) {
                        *v_i += g_i.mul_bigint(scalar.into_bigint());
                    }
                    black_box(v)
                })
            },
        );

        // Benchmark online GLV version
        group.bench_with_input(
            BenchmarkId::new("glv_online", size),
            &(&generators, &scalar, &initial_values),
            |b, &(generators, scalar, initial_values)| {
                b.iter(|| {
                    let mut v = initial_values.clone();
                    vector_scalar_mul_add_online(&mut v, generators, *scalar);
                    black_box(v)
                })
            },
        );

        // Benchmark precomputed GLV version (not counting precomputation time)
        let precomputed_data = VectorScalarMulData::new(&generators, scalar);
        group.bench_with_input(
            BenchmarkId::new("glv_precomputed", size),
            &(&precomputed_data, &initial_values),
            |b, &(data, initial_values)| {
                b.iter(|| {
                    let mut v = initial_values.clone();
                    vector_scalar_mul_add_precomputed(&mut v, data);
                    black_box(v)
                })
            },
        );

        // Benchmark precomputed GLV with precomputation time included
        group.bench_with_input(
            BenchmarkId::new("glv_precomputed_with_setup", size),
            &(&generators, &scalar, &initial_values),
            |b, &(generators, scalar, initial_values)| {
                b.iter_custom(|iters| {
                    let mut total_time = std::time::Duration::default();

                    for _ in 0..iters {
                        let mut v = initial_values.clone();

                        // Time includes both precomputation and multiplication
                        let start = Instant::now();
                        let data = VectorScalarMulData::new(generators, *scalar);
                        vector_scalar_mul_add_precomputed(&mut v, &data);
                        total_time += start.elapsed();

                        black_box(v);
                    }

                    total_time
                })
            },
        );

        // Benchmark the convenience function (which includes precomputation internally)
        group.bench_with_input(
            BenchmarkId::new("glv_convenience", size),
            &(&generators, &scalar, &initial_values),
            |b, &(generators, scalar, initial_values)| {
                b.iter(|| {
                    let mut v = initial_values.clone();
                    vector_scalar_mul_add(&mut v, generators, *scalar);
                    black_box(v)
                })
            },
        );

        group.finish();
    }
}

// Benchmark specifically for scenarios where the same scalar is used multiple times
fn bench_amortized_scalar_mul(c: &mut Criterion) {
    let mut rng = test_rng();

    const VECTOR_SIZE: usize = 1000;
    const NUM_APPLICATIONS: usize = 10; // Number of times to apply the same scalar

    // Generate test data
    let scalar = Fr::rand(&mut rng);

    // Generate multiple sets of generators (simulating different batches)
    let generator_batches: Vec<Vec<G2Projective>> = (0..NUM_APPLICATIONS)
        .map(|_| {
            (0..VECTOR_SIZE)
                .map(|_| G2Affine::rand(&mut rng).into_group())
                .collect()
        })
        .collect();

    // Initial values for each batch
    let initial_batches: Vec<Vec<G2Projective>> = (0..NUM_APPLICATIONS)
        .map(|_| {
            (0..VECTOR_SIZE)
                .map(|_| G2Affine::rand(&mut rng).into_group())
                .collect()
        })
        .collect();

    let mut group = c.benchmark_group("amortized_scalar_mul");

    // Benchmark naive approach for all batches
    group.bench_function("naive_all_batches", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for (generators, initial_values) in generator_batches.iter().zip(initial_batches.iter())
            {
                let mut v = initial_values.clone();
                for (v_i, g_i) in v.iter_mut().zip(generators.iter()) {
                    *v_i += g_i.mul_bigint(scalar.into_bigint());
                }
                results.push(v);
            }
            black_box(results)
        })
    });

    // Benchmark GLV online for all batches
    group.bench_function("glv_online_all_batches", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for (generators, initial_values) in generator_batches.iter().zip(initial_batches.iter())
            {
                let mut v = initial_values.clone();
                vector_scalar_mul_add_online(&mut v, generators, scalar);
                results.push(v);
            }
            black_box(results)
        })
    });

    // Benchmark GLV with separate precomputation for each batch
    let precomputed_datas: Vec<VectorScalarMulData> = generator_batches
        .iter()
        .map(|generators| VectorScalarMulData::new(generators, scalar))
        .collect();

    group.bench_function("glv_precomputed_all_batches", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for (data, initial_values) in precomputed_datas.iter().zip(initial_batches.iter()) {
                let mut v = initial_values.clone();
                vector_scalar_mul_add_precomputed(&mut v, data);
                results.push(v);
            }
            black_box(results)
        })
    });

    group.finish();
}

// Benchmark the new v[i] = scalar * v[i] + generators[i] operations
fn bench_vector_scalar_mul_v_add_g(c: &mut Criterion) {
    let mut rng = test_rng();

    // Test with different vector sizes
    let vector_sizes = [1000];

    for &size in &vector_sizes {
        // Generate test data
        let generators: Vec<G2Projective> = (0..size)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();
        let scalar = Fr::rand(&mut rng);

        // Initial values for the vector
        let initial_values: Vec<G2Projective> = (0..size)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();

        let mut group = c.benchmark_group(format!("vector_scalar_mul_v_add_g_{}", size));

        // Benchmark naive approach: v[i] = scalar * v[i] + generators[i]
        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &(&generators, &scalar, &initial_values),
            |b, &(generators, scalar, initial_values)| {
                b.iter(|| {
                    let mut v = initial_values.clone();
                    // Naive implementation: v[i] = scalar * v[i] + generators[i]
                    for (v_i, g_i) in v.iter_mut().zip(generators.iter()) {
                        *v_i = v_i.mul_bigint(scalar.into_bigint()) + g_i;
                    }
                    black_box(v)
                })
            },
        );

        // Benchmark online GLV version
        group.bench_with_input(
            BenchmarkId::new("glv_online", size),
            &(&generators, &scalar, &initial_values),
            |b, &(generators, scalar, initial_values)| {
                b.iter(|| {
                    let mut v = initial_values.clone();
                    vector_scalar_mul_v_add_g_online(&mut v, generators, *scalar);
                    black_box(v)
                })
            },
        );

        // Benchmark precomputed GLV version (not counting precomputation time)
        let precomputed_v_data = VectorScalarMulVData::new(scalar);
        group.bench_with_input(
            BenchmarkId::new("glv_precomputed", size),
            &(&precomputed_v_data, &generators, &initial_values),
            |b, &(data, generators, initial_values)| {
                b.iter(|| {
                    let mut v = initial_values.clone();
                    vector_scalar_mul_v_add_g_precomputed(&mut v, generators, data);
                    black_box(v)
                })
            },
        );

        // Benchmark precomputed GLV with precomputation time included
        group.bench_with_input(
            BenchmarkId::new("glv_precomputed_with_setup", size),
            &(&generators, &scalar, &initial_values),
            |b, &(generators, scalar, initial_values)| {
                b.iter_custom(|iters| {
                    let mut total_time = std::time::Duration::default();

                    for _ in 0..iters {
                        let mut v = initial_values.clone();

                        // Time includes both precomputation and multiplication
                        let start = Instant::now();
                        let data = VectorScalarMulVData::new(*scalar);
                        vector_scalar_mul_v_add_g_precomputed(&mut v, generators, &data);
                        total_time += start.elapsed();

                        black_box(v);
                    }

                    total_time
                })
            },
        );

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_vector_scalar_mul_add,
    bench_amortized_scalar_mul,
    bench_vector_scalar_mul_v_add_g
);
criterion_main!(benches);
