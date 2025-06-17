//! Optimized BN254 G2 scalar multiplication using 4D GLV decomposition
//!
//! This crate provides optimized scalar multiplication algorithms for BN254 G2
//! using 4-dimensional decomposition combining GLV and Frobenius endomorphisms.
//!
//! The main optimization reduces a 256-bit scalar multiplication to four ~66-bit
//! scalar multiplications, providing significant speedup for MSM operations.

pub mod constants;
pub mod decomp_2d;
pub mod decomp_4d;
pub mod dory_utils;
pub mod frobenius;
pub mod glv_two;
pub mod glv_two_new;

mod glv_four;
pub use glv_four::{
    fixed_base_vector_msm_g2, glv_four_precompute, glv_four_precompute_windowed,
    glv_four_precompute_windowed2_compact, glv_four_precompute_windowed2_signed,
    glv_four_scalar_mul, glv_four_scalar_mul_decomposed, glv_four_scalar_mul_online,
    glv_four_scalar_mul_windowed, glv_four_scalar_mul_windowed2_signed,
    glv_four_scalar_mul_windowed2_signed_decomposed, glv_four_scalar_mul_windowed_decomposed,
    shamir_glv_mul, shamir_glv_mul_precomputed, shamir_glv_mul_windowed,
    shamir_glv_mul_windowed2_signed, DecomposedScalar, FixedBasePrecomputedG2,
    PrecomputedShamirData, PrecomputedShamirTable, Windowed2CompactData, Windowed2CompactTable,
    Windowed2SignedData, Windowed2SignedTable, WindowedData, WindowedTable,
};

/// Re-export commonly used types
pub use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};

/// Re-export G1 2D GLV utilities
pub use glv_two::{
    fixed_base_vector_msm_g1,
    glv_two_precompute,
    glv_two_precompute_windowed2_signed,
    glv_two_scalar_mul,
    glv_two_scalar_mul_decomposed,
    glv_two_scalar_mul_online,
    glv_two_scalar_mul_windowed2_signed,
    // G1 vector scalar multiplication utilities
    vector_scalar_mul_add_g1,
    vector_scalar_mul_add_g1_online,
    vector_scalar_mul_add_g1_precomputed,
    vector_scalar_mul_v_add_g_g1_online,
    vector_scalar_mul_v_add_g_g1_precomputed,
    DecomposedScalar2D,
    FixedBasePrecomputedG1,
    PrecomputedShamir2Data,
    PrecomputedShamir2Table,
    VectorScalarMulG1Data,
    VectorScalarMulG1VData,
    Windowed2Signed2Data,
    Windowed2Signed2Table,
};

/// Re-export Dory utilities
pub use dory_utils::{
    vector_scalar_mul_add, vector_scalar_mul_add_online, vector_scalar_mul_add_precomputed,
    vector_scalar_mul_v_add_g_online, vector_scalar_mul_v_add_g_precomputed, VectorScalarMulData,
    VectorScalarMulVData,
};

/// Re-export Frobenius utilities
pub use frobenius::frobenius_psi_power_projective;

/// Re-export new G1 2D GLV utilities (temporary for testing)
pub use glv_two_new::{
    glv_two_scalar_mul_online as glv_two_scalar_mul_online_new,
    glv_two_precompute as glv_two_precompute_new,
    glv_two_scalar_mul as glv_two_scalar_mul_new,
    glv_two_precompute_windowed2_signed as glv_two_precompute_windowed2_signed_new,
    glv_two_scalar_mul_windowed2_signed as glv_two_scalar_mul_windowed2_signed_new,
};
