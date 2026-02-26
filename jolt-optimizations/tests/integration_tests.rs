use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::PrimeGroup;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{PrimeField, UniformRand};
use ark_std::{test_rng, One, Zero};

use jolt_optimizations::constants::get_bn254_frobenius_eigenvalue;
use jolt_optimizations::decomp_4d::decompose_scalar_4d;
use jolt_optimizations::frobenius::frobenius_psi_power_projective;

fn verify_decomposition(
    k: &Fr,
    coeffs: &[<Fr as PrimeField>::BigInt; 4],
    signs: &[bool; 4],
) -> bool {
    let lambda_psi = get_bn254_frobenius_eigenvalue();

    let mut reconstructed = Fr::zero();
    let lambda_powers = [
        Fr::one(),
        lambda_psi,
        lambda_psi * lambda_psi,
        lambda_psi * lambda_psi * lambda_psi,
    ];

    for i in 0..4 {
        let ki = Fr::from_bigint(coeffs[i]).unwrap();
        if signs[i] {
            reconstructed -= ki * lambda_powers[i];
        } else {
            reconstructed += ki * lambda_powers[i];
        }
    }

    *k == reconstructed
}

fn bigint_coeff_to_u128(c: &<Fr as PrimeField>::BigInt) -> u128 {
    (c.0[0] as u128) | ((c.0[1] as u128) << 64)
}

#[test]
fn test_sage_compatibility() {
    let test_cases: &[(&str, (u128, u128, u128, u128), (bool, bool, bool, bool))] = &[
        (
            "2ffdd975c07ece990ef2aeea9920c65dfc712bd1163e8a2f7c83a49c56d5a734",
            (
                88042004215725297833,
                15108385708047359372,
                3026787831446614349,
                1979941837268327954,
            ),
            (false, false, false, false),
        ),
        (
            "26c8ae8c1778176ea4e8513d9ed63d752bac8867fff3864fab091b5942dfae90",
            (
                11419926953705671942,
                7811088190336128973,
                1826629594471335422,
                22015774689864596198,
            ),
            (true, true, true, true),
        ),
        (
            "2f5fd543c668ecc4e3b0dce9056f6c98478980b48fb257d49b8c00d39d34b7a5",
            (
                35754355908556826287,
                20910459483802393987,
                1028669091245124392,
                28869705842701787973,
            ),
            (false, true, false, false),
        ),
        (
            "15a1e07d6b2dd8b556f5bf7635127ab136889ca0e5187c5f489a2567724f36f8",
            (
                19611214812899750558,
                21033706128297981334,
                11593548150491899665,
                35317176612977865430,
            ),
            (false, false, true, true),
        ),
        (
            "1bec1358f762c8c545f99b8bfb618df2fdb8ca7c44f2e2747c35bcf0a9459787",
            (
                7872138357851074524,
                13298440468603925221,
                27497847960248044950,
                14701329493939829561,
            ),
            (false, true, true, false),
        ),
    ];

    for (scalar_hex, expected_coeffs, expected_signs) in test_cases {
        let scalar_bytes = hex::decode(scalar_hex).unwrap();
        let scalar_fr = Fr::from_be_bytes_mod_order(&scalar_bytes);

        let (our_coeffs, our_signs) = decompose_scalar_4d(scalar_fr);

        assert_eq!(bigint_coeff_to_u128(&our_coeffs[0]), expected_coeffs.0);
        assert_eq!(bigint_coeff_to_u128(&our_coeffs[1]), expected_coeffs.1);
        assert_eq!(bigint_coeff_to_u128(&our_coeffs[2]), expected_coeffs.2);
        assert_eq!(bigint_coeff_to_u128(&our_coeffs[3]), expected_coeffs.3);

        assert_eq!(our_signs[0], expected_signs.0);
        assert_eq!(our_signs[1], expected_signs.1);
        assert_eq!(our_signs[2], expected_signs.2);
        assert_eq!(our_signs[3], expected_signs.3);

        assert!(
            verify_decomposition(&scalar_fr, &our_coeffs, &our_signs),
            "Algebraic check failed for scalar 0x{}",
            scalar_hex
        );

        // Verify point equation: k*P == k0*P + k1*ψ(P) + k2*ψ²(P) + k3*ψ³(P)
        let mut rng = test_rng();
        let p = G2Affine::rand(&mut rng).into_group();

        let k_times_p = p.mul_bigint(scalar_fr.into_bigint());

        let psi_powers: [G2Projective; 4] = std::array::from_fn(|i| {
            if i == 0 {
                p
            } else {
                frobenius_psi_power_projective(&p, i)
            }
        });

        let mut result = G2Projective::zero();
        for i in 0..4 {
            let ki_pi = psi_powers[i].mul_bigint(our_coeffs[i]);
            if our_signs[i] {
                result -= ki_pi;
            } else {
                result += ki_pi;
            }
        }

        assert_eq!(
            k_times_p, result,
            "Point equation failed for scalar 0x{}",
            scalar_hex
        );
    }
}

#[test]
fn test_4d_decomposition_on_points() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let k = Fr::rand(&mut rng);
        let p = G2Affine::generator()
            .into_group()
            .mul_bigint(Fr::rand(&mut rng).into_bigint())
            .into_affine()
            .into_group();

        let (coeffs, signs) = decompose_scalar_4d(k);

        assert!(
            verify_decomposition(&k, &coeffs, &signs),
            "Algebraic decomposition incorrect"
        );

        let k_times_p = p.mul_bigint(k.into_bigint());

        let psi_powers: [G2Projective; 4] = std::array::from_fn(|i| {
            if i == 0 {
                p
            } else {
                frobenius_psi_power_projective(&p, i)
            }
        });

        let mut result = G2Projective::zero();
        for i in 0..4 {
            let ki_pi = psi_powers[i].mul_bigint(coeffs[i]);
            if signs[i] {
                result -= ki_pi;
            } else {
                result += ki_pi;
            }
        }

        assert_eq!(k_times_p, result, "Point equation failed");
    }
}
