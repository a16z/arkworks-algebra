use ark_ff::fields::{Fp256, MontBackend, MontConfig};

/// Defines the parameters for the base field `Fq` of the BN254 curve.
#[derive(MontConfig)]
#[modulus = "21888242871839275222246405745257275088696311157297823662689037894645226208583"]
#[generator = "3"]
pub struct FqConfig;

/// The base field `Fq` of the BN254 curve.
pub type Fq = Fp256<MontBackend<FqConfig, 4>>;

pub const FQ_ONE: Fq = ark_ff::MontFp!("1");
pub const FQ_ZERO: Fq = ark_ff::MontFp!("0");
