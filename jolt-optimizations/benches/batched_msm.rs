use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_std::UniformRand;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jolt_optimizations::{msm_batched_bn254_tile_k, BatchedMsmConfig};
use rayon::prelude::*;

fn bench_batched_msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("batched_msm");
    group.sample_size(10);
    let mut rng = ark_std::test_rng();

    let n = 1 << 10;
    let m = 1 << 1;

    let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();

    let scalars_owned: Vec<Vec<Fr>> = (0..m)
        .map(|_| (0..n).map(|_| Fr::rand(&mut rng)).collect())
        .collect();

    let scalars_ref: Vec<&[Fr]> = scalars_owned.iter().map(|v| v.as_slice()).collect();

    let cfg = BatchedMsmConfig::default(); // GLV enabled by default

    group.bench_function("naive_parallel", |b| {
        b.iter(|| {
            black_box(
                scalars_owned
                    .par_iter()
                    .map(|scalars| G1Projective::msm(&bases, scalars).unwrap().into_affine())
                    .collect::<Vec<_>>(),
            )
        });
    });

    group.bench_function("batched_tile_k", |b| {
        b.iter(|| black_box(msm_batched_bn254_tile_k(&bases, &scalars_ref, &cfg)));
    });

    group.finish();
}

criterion_group!(benches, bench_batched_msm);
criterion_main!(benches);
