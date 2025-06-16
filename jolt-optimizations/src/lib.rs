//! Optimized BN254 G2 scalar multiplication using 4D GLV decomposition
//!
//! This crate provides optimized scalar multiplication algorithms for BN254 G2
//! using 4-dimensional decomposition combining GLV and Frobenius endomorphisms.
//!
//! The main optimization reduces a 256-bit scalar multiplication to four ~66-bit
//! scalar multiplications, providing significant speedup for MSM operations.

pub mod constants;
pub mod decomposition;
pub mod frobenius;

mod msm;
pub use msm::{
    msm_small_66bit, msm_small_66bit_precomputed, print_msm_profile, PrecomputedShamirData,
    PrecomputedShamirTable,
};

/// Re-export commonly used types
pub use ark_bn254::{Fr, G2Affine, G2Projective};
