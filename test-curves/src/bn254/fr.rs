use ark_ff::fields::{Fp256, MontBackend, MontConfig};
// use ark_ff_macros::{MontConfig, MontFp}; // Keep MontFp for constants

/// Defines the parameters for the scalar field `Fr` of the BN254 curve.
#[derive(MontConfig)]
#[modulus = "21888242871839275222246405745257275088548364400416034343698204186575808495617"]
#[generator = "5"]
// Define Small subgroup base/power if needed, otherwise macro defaults are fine
// #[small_subgroup_base = "3"]
// #[small_subgroup_power = "2"]
pub struct FrConfig;

/// The scalar field `Fr` of the BN254 curve.
pub type Fr = Fp256<MontBackend<FrConfig, 4>>;

pub const FR_ONE: Fr = ark_ff::MontFp!("1");
pub const FR_ZERO: Fr = ark_ff::MontFp!("0");

// For use in Tonelli-Shanks cube root finding.
// modulus - 1 = 3^s * t

// s = 2
pub const FR_THREE_ADICITY: usize = 2;
// t = 2432026985759919469138489527250808343172040488935114927077578242952867610624
pub const FR_TRACE_OF_MODULUS_MINUS_ONE: Fr =
    ark_ff::MontFp!("2432026985759919469138489527250808343172040488935114927077578242952867610624");

// (t - 1) / 3
pub const FR_TRACE_MINUS_ONE_DIV_BY_THREE: bool = true;
pub const FR_TRACE_PLUS_OR_MINUS_ONE_DIV_THREE: Fr =
    ark_ff::MontFp!("28586639823046163175750269447724013496311704975692526080984289203541");



