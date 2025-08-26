use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use ark_bn254::Fq;
use ark_ff::{One, UniformRand};
use ark_std::test_rng;

use jolt_optimizations::batched_expressions::verify_batched_expressions;
use jolt_optimizations::dory_fq12_utils::DoryState;

fn bench_dory_verification(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut group = c.benchmark_group("dory_verification");
    println!("benching...");
    for num_rounds in [15].iter() {
        let state = DoryState::random(*num_rounds);

        let alphas: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();
        let betas: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();
        let gammas: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();
        let s1_tildes: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();
        let s2_tildes: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();

        println!("Generating expressions for {} rounds...", num_rounds);
        let mut all_expressions = Vec::new();
        for round in 0..*num_rounds {
            println!("  Generating expressions for round {}...", round);
            let round_expressions = state.generate_round_expressions(
                round,
                alphas[round],
                betas[round],
                gammas[round],
                s1_tildes[round],
                s2_tildes[round],
            );
            all_expressions.extend(round_expressions);
        }
        println!("Total expressions generated: {}", all_expressions.len());
        
        // The expressions already have proper quotients computed from Fq12 values
        println!("Expressions with quotients ready!");

        // Benchmark naive computation (verifier computes everything in Fq12)
        group.bench_with_input(
            BenchmarkId::new("naive_fq12", num_rounds),
            num_rounds,
            |b, _| {
                b.iter(|| {
                    black_box(state.compute_all_rounds(
                        *num_rounds,
                        &alphas,
                        &betas,
                        &gammas,
                        &s1_tildes,
                        &s2_tildes,
                    ))
                })
            },
        );

        // Benchmark quotient-based verification (verifier just checks)
        let r = Fq::rand(&mut rng);
        let batch_gammas: Vec<Fq> = (0..all_expressions.len())
            .map(|_| Fq::rand(&mut rng))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("quotient_based", num_rounds),
            num_rounds,
            |b, _| {
                b.iter(|| {
                    black_box(verify_batched_expressions(
                        &all_expressions,
                        r,
                        &batch_gammas,
                    ))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dory_verification);
criterion_main!(benches);
