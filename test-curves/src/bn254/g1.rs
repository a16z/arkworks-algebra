use ark_ec::models::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_ec::CurveConfig;
use ark_ff::{AdditiveGroup, Field, MontFp, Zero};

use crate::bn254::{Fq, Fr}; // Assuming Fq is defined in fq.rs

#[derive(Clone, Default, PartialEq, Eq)]
pub struct G1Config;

impl CurveConfig for G1Config {
    type BaseField = Fq;
    type ScalarField = Fr;

    /// COFACTOR = 1
    const COFACTOR: &'static [u64] = &[1];

    /// COFACTOR_INV = COFACTOR^{-1} mod r = 1
    const COFACTOR_INV: Self::ScalarField = Fr::ONE;
}

impl SWCurveConfig for G1Config {
    /// COEFF_A = 0
    const COEFF_A: Self::BaseField = Fq::ZERO;

    /// COEFF_B = 3
    const COEFF_B: Self::BaseField = MontFp!("3");

    /// AFFINE_GENERATOR_COEFFS = (G1_GENERATOR_X, G1_GENERATOR_Y)
    const GENERATOR: Affine<Self> = Affine::new_unchecked(G1_GENERATOR_X, G1_GENERATOR_Y);

    #[inline(always)]
    fn mul_by_a(_base: Self::BaseField) -> Self::BaseField {
        // A is zero, avoids a multiply
        Self::BaseField::zero()
    }

    #[inline(always)]
    fn is_in_correct_subgroup_assuming_on_curve(_p: &Affine<Self>) -> bool {
        // The cofactor is 1, so any point on the curve is in the correct subgroup.
        true
    }
}

/// G1_GENERATOR_X = 1
pub const G1_GENERATOR_X: Fq = MontFp!("1");

/// G1_GENERATOR_Y = 2
pub const G1_GENERATOR_Y: Fq = MontFp!("2");

pub type G1Affine = Affine<G1Config>;
pub type G1Projective = Projective<G1Config>;
