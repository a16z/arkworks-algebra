use ark_bn254::{Bn254, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::{MillerLoopOutput, Pairing};
use ark_ff::One;
use ark_std::UniformRand;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rayon::prelude::*;

fn bench_multi_pairing_comparison(c: &mut Criterion) {
    let mut rng = ark_std::test_rng();
    let mut group = c.benchmark_group("multi_pairing_comparison");

    // Get number of threads for optimal chunking
    let num_threads = rayon::current_num_threads();

    for n_exp in [15] {
        let n = 1 << n_exp;
        group.throughput(Throughput::Elements(n as u64));

        let g1_points: Vec<G1Affine> = (0..n)
            .map(|_| G1Projective::rand(&mut rng).into())
            .collect();
        let g2_points: Vec<G2Affine> = (0..n)
            .map(|_| G2Projective::rand(&mut rng).into())
            .collect();

        // Precompute G1 and G2 prepared forms
        let g1_prepared: Vec<_> = g1_points
            .iter()
            .map(|p| <Bn254 as Pairing>::G1Prepared::from(p))
            .collect();
        let g2_prepared: Vec<_> = g2_points
            .iter()
            .map(|p| <Bn254 as Pairing>::G2Prepared::from(p))
            .collect();

        // Benchmark naive multi_miller_loop (uses internal chunking)
        group.bench_function(BenchmarkId::new("naive", n), |b| {
            b.iter(|| {
                let result = Bn254::multi_miller_loop(
                    g1_prepared.iter().cloned(),
                    g2_prepared.iter().cloned(),
                );
                black_box(Bn254::final_exponentiation(result))
            });
        });

        // Benchmark with outer parallelism - chunk size based on thread count
        let chunk_size = (n + num_threads - 1) / num_threads; // ceiling division
        println!("chunk {:?}", num_threads);
        if chunk_size > 0 {
            group.bench_function(
                BenchmarkId::new(format!("outer_parallel_threads_{}", num_threads), n),
                |b| {
                    b.iter(|| {
                        let ml_result: MillerLoopOutput<Bn254> = g1_prepared
                            .par_chunks(chunk_size)
                            .zip(g2_prepared.par_chunks(chunk_size))
                            .map(|(g1_chunk, g2_chunk)| {
                                Bn254::multi_miller_loop(
                                    g1_chunk.iter().cloned(),
                                    g2_chunk.iter().cloned(),
                                )
                            })
                            .reduce(
                                || MillerLoopOutput(<Bn254 as Pairing>::TargetField::one()),
                                |mut acc, val| {
                                    acc.0 *= val.0;
                                    acc
                                },
                            );
                        black_box(Bn254::final_exponentiation(ml_result))
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_multi_pairing_comparison,);
criterion_main!(benches);
