use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Instant;

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::PrimeGroup;
use ark_ec::{AdditiveGroup, AffineRepr};
use ark_ff::{PrimeField, UniformRand};
use ark_std::test_rng;

use jolt_optimizations::{
    vector_scalar_mul_add_g1, vector_scalar_mul_add_g1_online,
    vector_scalar_mul_add_g1_precomputed, vector_scalar_mul_v_add_g_g1_online,
    vector_scalar_mul_v_add_g_g1_precomputed, VectorScalarMulG1Data, VectorScalarMulG1VData,
};

fn bench_g1_vector_scalar_mul_add(c: &mut Criterion) {
    let mut rng = test_rng();

    // Test with different vector sizes
    let vector_sizes = [1000];

    for &size in &vector_sizes {
        // Generate test data
        let generators: Vec<G1Projective> = (0..size)
            .map(|_| G1Affine::rand(&mut rng).into_group())
            .collect();
        let scalar = Fr::rand(&mut rng);

        // Initial values for the vector
        let initial_values: Vec<G1Projective> = (0..size)
            .map(|_| G1Affine::rand(&mut rng).into_group())
            .collect();

        let mut group = c.benchmark_group(format!("g1_vector_scalar_mul_add_{}", size));

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
                    vector_scalar_mul_add_g1_online(&mut v, generators, *scalar);
                    black_box(v)
                })
            },
        );

        // Benchmark precomputed GLV version (not counting precomputation time)
        let precomputed_data = VectorScalarMulG1Data::new(&generators, scalar);
        group.bench_with_input(
            BenchmarkId::new("glv_precomputed", size),
            &(&precomputed_data, &initial_values),
            |b, &(data, initial_values)| {
                b.iter(|| {
                    let mut v = initial_values.clone();
                    vector_scalar_mul_add_g1_precomputed(&mut v, data);
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
                        let data = VectorScalarMulG1Data::new(generators, *scalar);
                        vector_scalar_mul_add_g1_precomputed(&mut v, &data);
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
                    vector_scalar_mul_add_g1(&mut v, generators, *scalar);
                    black_box(v)
                })
            },
        );

        group.finish();
    }
}

// Benchmark specifically for scenarios where the same scalar is used multiple times
fn bench_g1_amortized_scalar_mul(c: &mut Criterion) {
    let mut rng = test_rng();

    const VECTOR_SIZE: usize = 1000;
    const NUM_APPLICATIONS: usize = 10; // Number of times to apply the same scalar

    // Generate test data
    let scalar = Fr::rand(&mut rng);

    // Generate multiple sets of generators (simulating different batches)
    let generator_batches: Vec<Vec<G1Projective>> = (0..NUM_APPLICATIONS)
        .map(|_| {
            (0..VECTOR_SIZE)
                .map(|_| G1Affine::rand(&mut rng).into_group())
                .collect()
        })
        .collect();

    // Initial values for each batch
    let initial_batches: Vec<Vec<G1Projective>> = (0..NUM_APPLICATIONS)
        .map(|_| {
            (0..VECTOR_SIZE)
                .map(|_| G1Affine::rand(&mut rng).into_group())
                .collect()
        })
        .collect();

    let mut group = c.benchmark_group("g1_amortized_scalar_mul");

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
                vector_scalar_mul_add_g1_online(&mut v, generators, scalar);
                results.push(v);
            }
            black_box(results)
        })
    });

    // Benchmark GLV with separate precomputation for each batch
    let precomputed_datas: Vec<VectorScalarMulG1Data> = generator_batches
        .iter()
        .map(|generators| VectorScalarMulG1Data::new(generators, scalar))
        .collect();

    group.bench_function("glv_precomputed_all_batches", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for (data, initial_values) in precomputed_datas.iter().zip(initial_batches.iter()) {
                let mut v = initial_values.clone();
                vector_scalar_mul_add_g1_precomputed(&mut v, data);
                results.push(v);
            }
            black_box(results)
        })
    });

    group.finish();
}

// Benchmark the new G1 v[i] = scalar * v[i] + generators[i] operations
fn bench_g1_vector_scalar_mul_v_add_g(c: &mut Criterion) {
    let mut rng = test_rng();

    // Test with different vector sizes
    let vector_sizes = [1000];

    for &size in &vector_sizes {
        // Generate test data
        let generators: Vec<G1Projective> = (0..size)
            .map(|_| G1Affine::rand(&mut rng).into_group())
            .collect();
        let scalar = Fr::rand(&mut rng);

        // Initial values for the vector
        let initial_values: Vec<G1Projective> = (0..size)
            .map(|_| G1Affine::rand(&mut rng).into_group())
            .collect();

        let mut group = c.benchmark_group(format!("g1_vector_scalar_mul_v_add_g_{}", size));

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
                    vector_scalar_mul_v_add_g_g1_online(&mut v, generators, *scalar);
                    black_box(v)
                })
            },
        );

        // Benchmark precomputed GLV version (not counting precomputation time)
        let precomputed_v_data = VectorScalarMulG1VData::new(scalar);
        group.bench_with_input(
            BenchmarkId::new("glv_precomputed", size),
            &(&precomputed_v_data, &generators, &initial_values),
            |b, &(data, generators, initial_values)| {
                b.iter(|| {
                    let mut v = initial_values.clone();
                    vector_scalar_mul_v_add_g_g1_precomputed(&mut v, generators, data);
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
                        let data = VectorScalarMulG1VData::new(*scalar);
                        vector_scalar_mul_v_add_g_g1_precomputed(&mut v, generators, &data);
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
    bench_g1_vector_scalar_mul_add,
    bench_g1_amortized_scalar_mul,
    bench_g1_vector_scalar_mul_v_add_g
);
criterion_main!(benches);
