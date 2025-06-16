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

mod glv_four;
pub use glv_four::{
    glv_four_precompute, glv_four_scalar_mul, glv_four_scalar_mul_decomposed, glv_four_scalar_mul_online,
    DecomposedScalar, shamir_glv_mul, shamir_glv_mul_precomputed, PrecomputedShamirData, PrecomputedShamirTable,
};

/// Re-export commonly used types
pub use ark_bn254::{Fr, G2Affine, G2Projective};
