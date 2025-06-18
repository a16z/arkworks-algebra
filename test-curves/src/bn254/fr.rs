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
