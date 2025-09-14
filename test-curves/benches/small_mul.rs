use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use ark_test_curves::bn254::{Fr, FrConfig};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

// Hack: copy over the helper functions from the Montgomery backend to be benched

fn mul_small_bench(c: &mut Criterion) {
    const SAMPLES: usize = 1000;
    // Use a fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(0u64);

    let a_s = (0..SAMPLES)
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>();
    let a_limbs_s = a_s.iter().map(|a| a.0.0).collect::<Vec<_>>();

    let b_u64_s = (0..SAMPLES)
        .map(|_| rng.gen::<u64>())
        .collect::<Vec<_>>();
    // Convert u64 to Fr for standard multiplication benchmark
    let b_fr_s = b_u64_s.iter().map(|&b| Fr::from(b)).collect::<Vec<_>>();

    let b_u64_as_u128_s = b_u64_s.iter().map(|&b| b as u128).collect::<Vec<_>>();

    let b_i64_s = (0..SAMPLES)
        .map(|_| rng.gen::<i64>())
        .collect::<Vec<_>>();

    let b_u128_s = (0..SAMPLES)
        .map(|_| rng.gen::<u128>())
        .collect::<Vec<_>>();

    let b_i128_s = (0..SAMPLES)
        .map(|_| rng.gen::<i128>())
        .collect::<Vec<_>>();

    // Generate another set of random Fr elements for addition
    let c_s = (0..SAMPLES)
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("Fr Arithmetic Comparison");

    // group.bench_function("Addition (Fr + Fr)", |b| {
    //     let mut i = 0usize;
    //     b.iter_batched_ref(
    //         || {
    //             let pair = (a_s[i], c_s[i]);
    //             i += 1; if i == SAMPLES { i = 0; }
    //             pair
    //         },
    //         |pair| {
    //             let (a, c) = *pair;
    //             let a = criterion::black_box(a);
    //             let c = criterion::black_box(c);
    //             criterion::black_box(a + c)
    //         },
    //         BatchSize::SmallInput,
    //     )
    // });

    group.bench_function("mul_u64", |b| {
        let mut i = 0usize;
        b.iter_batched_ref(
            || {
                let pair = (a_s[i], b_u64_s[i]);
                i += 1; if i == SAMPLES { i = 0; }
                pair
            },
            |pair| {
                let (a, bu) = *pair;
                let a = criterion::black_box(a);
                let bu = criterion::black_box(bu);
                criterion::black_box(a.mul_u64(bu))
            },
            BatchSize::SmallInput,
        )
    });

    // group.bench_function("mul_u64_new", |b| {
    //     let mut i = 0usize;
    //     b.iter_batched_ref(
    //         || {
    //             let pair = (a_s[i], b_u64_s[i]);
    //             i += 1; if i == SAMPLES { i = 0; }
    //             pair
    //         },
    //         |pair| {
    //             let (a, bu) = *pair;
    //             let a = criterion::black_box(a);
    //             let bu = criterion::black_box(bu);
    //             criterion::black_box(a.mul_u64_new(bu))
    //         },
    //         BatchSize::SmallInput,
    //     )
    // });

    // group.bench_function("mul_i64", |b| {
    //     let mut i = 0usize;
    //     b.iter_batched_ref(
    //         || {
    //             let pair = (a_s[i], b_i64_s[i]);
    //             i += 1; if i == SAMPLES { i = 0; }
    //             pair
    //         },
    //         |pair| {
    //             let (a, bi) = *pair;
    //             let a = criterion::black_box(a);
    //             let bi = criterion::black_box(bi);
    //             criterion::black_box(a.mul_i64(bi))
    //         },
    //         BatchSize::SmallInput,
    //     )
    // });

    // Note: results might be worse than in real applications due to branch prediction being wrong
    // 50% of the time
    // group.bench_function("mul_u128", |b| {
    //     let mut i = 0usize;
    //     b.iter_batched(
    //         || {
    //             let pair = (a_s[i], b_u128_s[i]);
    //             i += 1; if i == SAMPLES { i = 0; }
    //             pair
    //         },
    //         |(a, bu)| {
    //             let a = criterion::black_box(a);
    //             let bu = criterion::black_box(bu);
    //             criterion::black_box(a.mul_u128(bu))
    //         },
    //         BatchSize::SmallInput,
    //     )
    // });

    // group.bench_function("mul_i128", |b| {
    //     let mut i = 0usize;
    //     b.iter_batched(
    //         || {
    //             let pair = (a_s[i], b_i128_s[i]);
    //             i += 1; if i == SAMPLES { i = 0; }
    //             pair
    //         },
    //         |(a, bi)| {
    //             let a = criterion::black_box(a);
    //             let bi = criterion::black_box(bi);
    //             criterion::black_box(a.mul_i128(bi))
    //         },
    //         BatchSize::SmallInput,
    //     )
    // });

    group.bench_function("standard mul (Fr * Fr)", |b| {
        let mut i = 0usize;
        b.iter_batched_ref(
            || {
                let pair = (a_s[i], b_fr_s[i]);
                i += 1; if i == SAMPLES { i = 0; }
                pair
            },
            |pair| {
                let (a, bf) = *pair;
                let a = criterion::black_box(a);
                let bf = criterion::black_box(bf);
                criterion::black_box(a * bf)
            },
            BatchSize::SmallInput,
        )
    });

    // // Benchmark mul_u128 specifically with inputs known to fit in u64
    // group.bench_function("mul_u128 (u64 inputs)", |b| {
    //     let mut i = 0usize;
    //     b.iter_batched(
    //         || {
    //             let pair = (a_s[i], b_u64_as_u128_s[i]);
    //             i += 1; if i == SAMPLES { i = 0; }
    //             pair
    //         },
    //         |(a, bu)| {
    //             let a = criterion::black_box(a);
    //             let bu = criterion::black_box(bu);
    //             criterion::black_box(a.mul_u128(bu))
    //         },
    //         BatchSize::SmallInput,
    //     )
    // });

    // Benchmark the auxiliary function directly (assuming it's made public)
    // Note: Requires mul_u128_aux to be pub in montgomery_backend.rs
    // Need to import it if not already done via wildcard/specific import
    // Let's assume it's accessible via a_s[i].mul_u128_aux(...) for now
    // group.bench_function("mul_u128_aux (u128 inputs)", |b| {
    //     let mut i = 0usize;
    //     b.iter_batched_ref(
    //         || {
    //             let pair = (a_s[i], b_u128_s[i]);
    //             i += 1; if i == SAMPLES { i = 0; }
    //             pair
    //         },
    //         |pair| {
    //             let (a, bu) = *pair;
    //             let a = criterion::black_box(a);
    //             let bu = criterion::black_box(bu);
    //             criterion::black_box(a.mul_u128_aux(bu))
    //         },
    //         BatchSize::SmallInput,
    //     )
    // });

    group.finish();
}

criterion_group!(benches, mul_small_bench);
criterion_main!(benches); 