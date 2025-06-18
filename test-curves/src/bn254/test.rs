#![allow(unused_imports)]
use ark_ec::{
    models::short_weierstrass::SWCurveConfig, // Keep this as G1 is SW
    pairing::Pairing,
    AffineRepr,
    CurveGroup,
    PrimeGroup,
};
use ark_ff::{Field, One, UniformRand, Zero};
use ark_std::{rand::Rng, test_rng};

// Add imports for the newly defined types
use crate::bn254::{Fq, FqConfig, Fr, FrConfig, G1Affine, G1Projective};

use ark_algebra_test_templates::*;
use ark_std::ops::{AddAssign, MulAssign, SubAssign};

test_field!(fr; Fr; mont_prime_field);
// Uncomment Fq test
test_field!(fq; Fq; mont_prime_field);

// Uncomment G1 test for Short Weierstrass
test_group!(g1; G1Projective; sw);

// Add other tests for G2, Pairing etc. as needed
