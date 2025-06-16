use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::{AffineRepr, Group};
use ark_ff::{PrimeField, UniformRand};
use ark_std::test_rng;

use jolt_optimizations::decomposition::{decompose_scalar_table_based, fr_to_bigint, u128_to_fr};
use jolt_optimizations::frobenius::frobenius_psi_power_projective;
use jolt_optimizations::{msm_small_66bit, msm_small_66bit_precomputed, PrecomputedShamirData};

/// Pre-computed decomposition data for efficient scalar multiplication
#[derive(Clone)]
pub struct PrecomputedDecomposition {
    pub k_bigint: [<Fr as PrimeField>::BigInt; 4], // Small scalars as arkworks BigInt
    pub signs: [bool; 4],                          // Sign flags
}

impl PrecomputedDecomposition {
    pub fn new(coeffs: &[u128; 4], signs: &[bool; 4]) -> Self {
        let k_bigint = [
            u128_to_fr(coeffs[0]).into_bigint(),
            u128_to_fr(coeffs[1]).into_bigint(),
            u128_to_fr(coeffs[2]).into_bigint(),
            u128_to_fr(coeffs[3]).into_bigint(),
        ];

        Self {
            k_bigint,
            signs: *signs,
        }
    }
}

/// Precomputed Frobenius powers for all points
pub struct PrecomputedFrobeniusPowers {
    pub powers: Vec<[G2Projective; 4]>, // [P, ψ(P), ψ²(P), ψ³(P)] for each point
}

impl PrecomputedFrobeniusPowers {
    pub fn new(points: &[G2Projective]) -> Self {
        let powers = points
            .par_iter()
            .map(|point| {
                [
                    *point,
                    frobenius_psi_power_projective(point, 1),
                    frobenius_psi_power_projective(point, 2),
                    frobenius_psi_power_projective(point, 3),
                ]
            })
            .collect();

        Self { powers }
    }
}

/// MSM-based scalar multiplication using 4D decomposition  
fn scalar_mul_4d_msm(
    frobenius_powers: &[G2Projective; 4],
    precomputed: &PrecomputedDecomposition,
) -> G2Projective {
    msm_small_66bit(frobenius_powers, &precomputed.k_bigint, &precomputed.signs)
}

fn bench_scalar_multiplication(c: &mut Criterion) {
    let mut rng = test_rng();

    // Fix a random scalar for all tests
    let scalar = Fr::rand(&mut rng);
    let scalar_bigint = fr_to_bigint(scalar);
    let (coeffs, signs) = decompose_scalar_table_based(&scalar_bigint);
    let precomputed = PrecomputedDecomposition::new(&coeffs, &signs);

    const NUM_TESTS: usize = 10000;

    // Generate random points
    let points: Vec<G2Projective> = (0..NUM_TESTS)
        .map(|_| G2Affine::rand(&mut rng).into_group())
        .collect();

    let frobenius_powers = PrecomputedFrobeniusPowers::new(&points);
    let shamir_data = PrecomputedShamirData::new(&points);

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
        &shamir_data,
        |b, shamir_data| {
            b.iter(|| {
                shamir_data
                    .shamir_tables
                    .par_iter()
                    .map(|shamir_table| {
                        black_box(msm_small_66bit_precomputed(
                            shamir_table,
                            &precomputed.k_bigint,
                            &precomputed.signs,
                        ))
                    })
                    .collect::<Vec<G2Projective>>()
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("4d_online", NUM_TESTS),
        &frobenius_powers,
        |b, frobenius_powers| {
            b.iter(|| {
                frobenius_powers
                    .powers
                    .par_iter()
                    .map(|powers| black_box(scalar_mul_4d_msm(powers, &precomputed)))
                    .collect::<Vec<G2Projective>>()
            })
        },
    );

    group.finish();
}

criterion_group!(benches, bench_scalar_multiplication);
criterion_main!(benches);
