use ark_ff::{MontConfig, UniformRand};
use ark_test_curves::bn254::{Fr, FrConfig};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_no_carry_opt(c: &mut Criterion) {
    let mut group = c.benchmark_group("bn254_fr_mul_no_carry_opt");

    // Configure the benchmark
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(10));

    let mut rng = ark_std::test_rng();

    // Pre-generate test data
    let test_pairs: Vec<(Fr, Fr)> = (0..100_000)
        .map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)))
        .collect();

    group.bench_function(BenchmarkId::new("no_carry_opt", "10k_pairs"), |b| {
        b.iter(|| {
            for (a, b) in &test_pairs {
                let mut result = *a;
                <FrConfig as MontConfig<4>>::mul_assign_no_carry_opt(&mut result, b);
                black_box(result);
            }
        });
    });

    group.finish();
}

fn bench_standard_cios(c: &mut Criterion) {
    let mut group = c.benchmark_group("bn254_fr_mul_standard_cios");

    // Configure the benchmark
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(10));

    let mut rng = ark_std::test_rng();

    // Pre-generate test data
    let test_pairs: Vec<(Fr, Fr)> = (0..100_000)
        .map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)))
        .collect();

    group.bench_function(BenchmarkId::new("standard_cios", "10k_pairs"), |b| {
        b.iter(|| {
            for (a, b) in &test_pairs {
                let mut result = *a;
                <FrConfig as MontConfig<4>>::mul_assign_standard_cios(&mut result, b);
                black_box(result);
            }
        });
    });

    group.finish();
}

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("bn254_fr_mul_comparison");

    // Configure the benchmark for statistical comparison
    group.sample_size(200);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.significance_level(0.05); // 95% confidence level for p-test

    let mut rng = ark_std::test_rng();

    // Pre-generate test data - use same data for both benchmarks
    let test_pairs: Vec<(Fr, Fr)> = (0..100_000)
        .map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)))
        .collect();

    group.bench_function("no_carry_opt", |b| {
        b.iter(|| {
            for (a, b) in &test_pairs {
                let mut result = *a;
                <FrConfig as MontConfig<4>>::mul_assign_no_carry_opt(&mut result, b);
                black_box(result);
            }
        });
    });

    group.bench_function("standard_cios", |b| {
        b.iter(|| {
            for (a, b) in &test_pairs {
                let mut result = *a;
                <FrConfig as MontConfig<4>>::mul_assign_standard_cios(&mut result, b);
                black_box(result);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_no_carry_opt,
    bench_standard_cios,
    bench_comparison
);
criterion_main!(benches);
