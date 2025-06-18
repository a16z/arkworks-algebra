use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use ark_test_curves::bn254::{Fr, FrConfig};
use criterion::{criterion_group, criterion_main, Criterion};

// Hack: copy over the helper functions from the Montgomery backend to be benched

fn mul_small_bench(c: &mut Criterion) {
    const SAMPLES: usize = 1000;
    // Use a fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(0u64);

    let a_s = (0..SAMPLES).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
    let a_limbs_s = a_s.iter().map(|a| a.0 .0).collect::<Vec<_>>();

    let b_u64_s = (0..SAMPLES).map(|_| rng.gen::<u64>()).collect::<Vec<_>>();
    // Convert u64 to Fr for standard multiplication benchmark
    let b_fr_s = b_u64_s.iter().map(|&b| Fr::from(b)).collect::<Vec<_>>();

    let b_u64_as_u128_s = b_u64_s.iter().map(|&b| b as u128).collect::<Vec<_>>();

    let b_i64_s = (0..SAMPLES).map(|_| rng.gen::<i64>()).collect::<Vec<_>>();

    let b_u128_s = (0..SAMPLES).map(|_| rng.gen::<u128>()).collect::<Vec<_>>();

    let b_i128_s = (0..SAMPLES).map(|_| rng.gen::<i128>()).collect::<Vec<_>>();

    // Generate another set of random Fr elements for addition
    let c_s = (0..SAMPLES).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

    let mut group = c.benchmark_group("Fr Arithmetic Comparison");

    group.bench_function("mul_u64", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i].mul_u64(b_u64_s[i]))
        })
    });

    group.bench_function("mul_i64", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i].mul_i64(b_i64_s[i]))
        })
    });

    // Note: results might be worse than in real applications due to branch prediction being wrong
    // 50% of the time
    group.bench_function("mul_u128", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i].mul_u128(b_u128_s[i]))
        })
    });

    group.bench_function("mul_i128", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i].mul_i128(b_i128_s[i]))
        })
    });

    group.bench_function("standard mul (Fr * Fr)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i] * b_fr_s[i])
        })
    });

    // Benchmark mul_u128 specifically with inputs known to fit in u64
    group.bench_function("mul_u128 (u64 inputs)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            // Call mul_u128 but provide a u64 input cast to u128
            criterion::black_box(a_s[i].mul_u128(b_u64_as_u128_s[i]))
        })
    });

    // Benchmark the auxiliary function directly (assuming it's made public)
    // Note: Requires mul_u128_aux to be pub in montgomery_backend.rs
    // Need to import it if not already done via wildcard/specific import
    // Let's assume it's accessible via a_s[i].mul_u128_aux(...) for now
    group.bench_function("mul_u128_aux (u128 inputs)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i].mul_u128_aux(b_u128_s[i]))
        })
    });

    group.bench_function("Addition (Fr + Fr)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i] + c_s[i])
        })
    });

    group.finish();
}

criterion_group!(benches, mul_small_bench);
criterion_main!(benches);
