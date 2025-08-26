use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use ark_bn254::Fq;
use ark_ff::{One, UniformRand};
use ark_std::test_rng;

use jolt_optimizations::batched_expressions::verify_batched_expressions;
use jolt_optimizations::dory_fq12_utils::DoryState;
use jolt_optimizations::g_coeffs;

fn bench_dory_verification(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut group = c.benchmark_group("dory_verification");
    println!("benching...");
    // Test different numbers of rounds
    for num_rounds in [15].iter() {
        let state = DoryState::random(*num_rounds);

        // Generate random challenges for each round
        let alphas: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();
        let betas: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();
        let gammas: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();
        let s1_tildes: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();
        let s2_tildes: Vec<Fq> = (0..*num_rounds).map(|_| Fq::rand(&mut rng)).collect();

        // Generate all expressions for all rounds using DoryState method
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

        // For benchmarking, use dummy quotients since real computation requires Fq12 values
        // In practice, the prover would compute these from the actual Fq12 elements
        println!("Setting up quotients for benchmarking...");
        let g = g_coeffs();
        let mut expressions_with_quotients = all_expressions.clone();
        for expression in expressions_with_quotients.iter_mut() {
            // Use a dummy quotient for benchmarking
            // Real implementation would compute from Fq12 values
            expression.quotient = Some(vec![Fq::one(); 10]);
        }
        println!("Quotients ready!");

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
        let batch_gammas: Vec<Fq> = (0..expressions_with_quotients.len())
            .map(|_| Fq::rand(&mut rng))
            .collect();
        let g_array: [Fq; 13] = g.clone().try_into().unwrap();

        group.bench_with_input(
            BenchmarkId::new("quotient_based", num_rounds),
            num_rounds,
            |b, _| {
                b.iter(|| {
                    black_box(verify_batched_expressions(
                        &expressions_with_quotients,
                        r,
                        &batch_gammas,
                        &g_array,
                    ))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dory_verification);
criterion_main!(benches);
