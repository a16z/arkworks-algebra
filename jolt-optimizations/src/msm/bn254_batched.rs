//! Tile-K Batched MSM for BN254 G1 with GLV endomorphism
//!
//! Computes M MSMs of size N with shared bases using K-way tiling
//! for high throughput on CPU.

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{AdditiveGroup, BigInteger, PrimeField};
use ark_std::Zero;
use rayon::prelude::*;

use crate::decomp_2d::{decompose_scalar_2d, glv_endomorphism};

/// Configuration for batched MSM computation
#[derive(Clone, Debug)]
pub struct BatchedMsmConfig {
    /// Window size (bits) for signed-window digits. None = auto-select based on N.
    pub window_bits: Option<usize>,
    /// Tile size K: number of MSMs fused per pass.
    pub tile_k: usize,
    /// Column chunk size for streaming (in bases).
    pub col_chunk: usize,
    /// Enable mixed-add path (normalize touched bases per tile/window).
    pub mixed_add: bool,
    /// Enable GLV split.
    pub use_glv: bool,
}

impl Default for BatchedMsmConfig {
    fn default() -> Self {
        Self {
            window_bits: Some(7), // Auto-select
            tile_k: 64,
            col_chunk: 1 << 8,
            mixed_add: true,
            use_glv: true,
        }
    }
}

/// Choose optimal window size based on number of bases
fn choose_window_size(n: usize) -> usize {
    let lg = (usize::BITS - (n.max(1) as u32).leading_zeros()) as usize;
    let w = lg.saturating_sub(1);
    w.clamp(4, 16) // BN254 CPU sweet spot: 4-16 bits
}

/// CSR (Compressed Sparse Row) structure for bucket contributions
struct CsrBuckets {
    /// Row pointers: row_ptr[k][b] = start index in col_idx/sign_bits for MSM k, bucket b
    row_ptr: Vec<Vec<usize>>,
    /// Column indices (base indices)
    col_idx: Vec<u32>,
    /// Sign bits: 0 = positive, 1 = negative
    sign_bits: Vec<u8>,
    /// Highest used bucket per k (for trimming empty buckets in fold)
    b_hi: Vec<usize>,
}

/// Extract unsigned digit at window position t for scalar s
/// Returns value in range [0, 2^w - 1]
#[inline]
fn extract_digit_u(s: &<Fr as PrimeField>::BigInt, t: usize, w: usize) -> u16 {
    let bit_pos = t * w;
    let num_bits = s.num_bits() as usize;
    if bit_pos >= num_bits {
        return 0;
    }

    let mut digit = 0u16;
    let end = core::cmp::min(w, num_bits - bit_pos);
    for i in 0..end {
        if s.get_bit(bit_pos + i) {
            digit |= 1 << i;
        }
    }
    digit
}

/// Precomputed GLV decomposition for a tile
struct GlvSplit {
    mag0: Vec<<Fr as PrimeField>::BigInt>,
    mag1: Vec<<Fr as PrimeField>::BigInt>,
    sign0_pos: Vec<bool>,
    sign1_pos: Vec<bool>,
}

/// Precompute GLV decomposition for entire tile
fn precompute_glv_tile(scalars: &[Fr], use_glv: bool) -> GlvSplit {
    if !use_glv {
        GlvSplit {
            mag0: scalars.iter().map(|s| s.into_bigint()).collect(),
            mag1: Vec::new(),
            sign0_pos: vec![true; scalars.len()],
            sign1_pos: Vec::new(),
        }
    } else {
        let n = scalars.len();
        let mut mag0 = Vec::with_capacity(n);
        let mut mag1 = Vec::with_capacity(n);
        let mut s0p = Vec::with_capacity(n);
        let mut s1p = Vec::with_capacity(n);

        for s in scalars.iter() {
            let (coeffs, signs) = decompose_scalar_2d(*s);
            mag0.push(coeffs[0]);
            mag1.push(coeffs[1]);
            s0p.push(signs[0]);
            s1p.push(signs[1]);
        }

        GlvSplit {
            mag0,
            mag1,
            sign0_pos: s0p,
            sign1_pos: s1p,
        }
    }
}

/// All digits for all windows: [num_windows][N]
struct AllWindowDigits {
    // u0[window_idx][scalar_idx]
    u0: Vec<Vec<u16>>,
    // u1[window_idx][scalar_idx] (if GLV)
    u1: Vec<Vec<u16>>,
}

/// Precompute ALL window digits at once (not per-window)
fn compute_all_window_digits(
    glv: &GlvSplit,
    num_windows: usize,
    window_bits: usize,
    use_glv: bool,
) -> AllWindowDigits {
    let n = glv.mag0.len();

    // Precompute all windows for u0
    let u0: Vec<Vec<u16>> = (0..num_windows)
        .into_par_iter()
        .map(|w_idx| {
            glv.mag0
                .iter()
                .map(|m| extract_digit_u(m, w_idx, window_bits))
                .collect()
        })
        .collect();

    let u1: Vec<Vec<u16>> = if use_glv {
        (0..num_windows)
            .into_par_iter()
            .map(|w_idx| {
                glv.mag1
                    .iter()
                    .map(|m| extract_digit_u(m, w_idx, window_bits))
                    .collect()
            })
            .collect()
    } else {
        vec![vec![]; num_windows]
    };

    AllWindowDigits { u0, u1 }
}

/// Build CSR structure using precomputed digits
fn build_csr_from_digits(
    all_digits_tile: &[&AllWindowDigits], // K_eff entries
    glv_splits: &[&GlvSplit],
    window_idx: usize,
    col_start: usize,
    col_end: usize,
    use_glv: bool,
    b_max: usize,
) -> CsrBuckets {
    let k_eff = all_digits_tile.len();
    let n = all_digits_tile[0].u0[0].len();

    // Phase A: Histogram counts
    let counts: Vec<Vec<usize>> = (0..k_eff)
        .into_par_iter()
        .map(|k| {
            let all_digits = all_digits_tile[k];
            let mut cnt = vec![0usize; b_max];

            for col in col_start..col_end {
                let (scalar_idx, is_phi) = if use_glv {
                    if col < n {
                        (col, false)
                    } else {
                        (col - n, true)
                    }
                } else {
                    (col, false)
                };

                if scalar_idx >= n {
                    continue;
                }

                let u = if is_phi {
                    all_digits.u1[window_idx][scalar_idx]
                } else {
                    all_digits.u0[window_idx][scalar_idx]
                } as usize;

                debug_assert!(u <= b_max, "digit u={} out of range (max={})", u, b_max);
                if u != 0 {
                    let bucket = u - 1;
                    debug_assert!(bucket < b_max, "bucket={} >= b_max={}", bucket, b_max);
                    cnt[bucket] += 1;
                }
            }
            cnt
        })
        .collect();

    // Phase B: Exclusive scan → row_ptr and compute b_hi
    let mut row_ptr = vec![vec![0usize; b_max + 1]; k_eff];
    let mut b_hi = vec![0usize; k_eff];
    let mut total_nnz = 0usize;

    for k in 0..k_eff {
        for b in 0..b_max {
            row_ptr[k][b] = total_nnz;
            total_nnz += counts[k][b];
            if counts[k][b] > 0 {
                b_hi[k] = b + 1; // Track highest non-empty bucket
            }
        }
        row_ptr[k][b_max] = total_nnz;
    }

    // Phase C: Scatter
    let mut col_idx = vec![0u32; total_nnz];
    let mut sign_bits = vec![0u8; total_nnz];
    let mut write_pos = row_ptr.clone();

    for k in 0..k_eff {
        let all_digits = all_digits_tile[k];
        let glv = glv_splits[k];

        for col in col_start..col_end {
            let (scalar_idx, is_phi) = if use_glv {
                if col < n {
                    (col, false)
                } else {
                    (col - n, true)
                }
            } else {
                (col, false)
            };

            if scalar_idx >= n {
                continue;
            }

            let (u, sign_pos) = if is_phi {
                (
                    all_digits.u1[window_idx][scalar_idx] as usize,
                    glv.sign1_pos[scalar_idx],
                )
            } else {
                (
                    all_digits.u0[window_idx][scalar_idx] as usize,
                    glv.sign0_pos[scalar_idx],
                )
            };

            if u != 0 {
                let bucket = u - 1;
                let pos = write_pos[k][bucket];
                col_idx[pos] = col as u32;
                sign_bits[pos] = (!sign_pos) as u8;
                write_pos[k][bucket] += 1;
            }
        }
    }

    // Debug assertions
    #[cfg(debug_assertions)]
    {
        for k in 0..k_eff {
            for b in 0..b_max {
                debug_assert!(
                    row_ptr[k][b] <= row_ptr[k][b + 1],
                    "Row pointers not monotonic: k={}, b={}",
                    k,
                    b
                );
            }

            let start_offset = if k == 0 { 0 } else { row_ptr[k - 1][b_max] };
            let end_offset = row_ptr[k][b_max];
            let total_nnz_k = end_offset - start_offset;
            let sum_counts: usize = counts[k].iter().sum();
            debug_assert_eq!(
                total_nnz_k, sum_counts,
                "NNZ mismatch for k={}: total_nnz={} != sum_counts={}",
                k, total_nnz_k, sum_counts
            );
        }

        for i in 0..col_idx.len() {
            debug_assert!(
                col_idx[i] >= col_start as u32 && col_idx[i] < col_end as u32,
                "col_idx[{}]={} out of bounds [{}, {})",
                i,
                col_idx[i],
                col_start,
                col_end
            );
            debug_assert!(
                sign_bits[i] <= 1,
                "sign_bits[{}]={} not in {{0,1}}",
                i,
                sign_bits[i]
            );
        }
    }

    CsrBuckets {
        row_ptr,
        col_idx,
        sign_bits,
        b_hi,
    }
}

/// Old build_csr_for_window kept for compatibility with tests
fn build_csr_for_window(
    scalars: &[&[Fr]], // K_eff × N
    window_idx: usize,
    window_bits: usize,
    col_start: usize,
    col_end: usize,
    use_glv: bool,
) -> CsrBuckets {
    let k_eff = scalars.len();
    // For unsigned fixed-window, buckets = 1..(2^w - 1)
    let b_max = (1usize << window_bits) - 1;

    // -------- Phase A: histogram counts --------
    let counts: Vec<Vec<usize>> = (0..k_eff)
        .into_par_iter()
        .map(|k| {
            let n = scalars[k].len();
            let mut cnt = vec![0usize; b_max]; // index 0..b_max-1 corresponds to u-1
            for col in col_start..col_end {
                // Map logical column → (scalar index, is_phi branch)
                let (scalar_idx, is_phi) = if use_glv {
                    if col < n {
                        (col, false)
                    } else {
                        (col - n, true)
                    }
                } else {
                    (col, false)
                };

                if scalar_idx >= n {
                    continue;
                }

                // GLV split once per (k, scalar_idx)
                if use_glv {
                    let (coeffs, signs) = decompose_scalar_2d(scalars[k][scalar_idx]);
                    let (mag, _sign_pos) = if is_phi {
                        (&coeffs[1], signs[1])
                    } else {
                        (&coeffs[0], signs[0])
                    };
                    let u = extract_digit_u(mag, window_idx, window_bits) as usize;
                    debug_assert!(
                        u < (1 << window_bits),
                        "digit u={} >= 2^w={}",
                        u,
                        1 << window_bits
                    );
                    if u != 0 {
                        let bucket = u - 1; // map 1..(2^w-1) → 0..b_max-1
                        debug_assert!(bucket < b_max, "bucket={} >= b_max={}", bucket, b_max);
                        cnt[bucket] += 1;
                    }
                } else {
                    let mag = &scalars[k][scalar_idx].into_bigint();
                    let u = extract_digit_u(mag, window_idx, window_bits) as usize;
                    debug_assert!(
                        u < (1 << window_bits),
                        "digit u={} >= 2^w={}",
                        u,
                        1 << window_bits
                    );
                    if u != 0 {
                        let bucket = u - 1;
                        debug_assert!(bucket < b_max, "bucket={} >= b_max={}", bucket, b_max);
                        cnt[bucket] += 1;
                    }
                }
            }
            cnt
        })
        .collect();

    // -------- Phase B: exclusive scans → row_ptr and compute b_hi --------
    let mut row_ptr = vec![vec![0usize; b_max + 1]; k_eff];
    let mut b_hi = vec![0usize; k_eff];
    let mut total_nnz = 0usize;
    for k in 0..k_eff {
        for b in 0..b_max {
            row_ptr[k][b] = total_nnz;
            total_nnz += counts[k][b];
            if counts[k][b] > 0 {
                b_hi[k] = b + 1;
            }
        }
        row_ptr[k][b_max] = total_nnz;
    }

    // -------- Phase C: scatter (no atomics; sequential here, parallelize by thread-local if you want) --------
    let mut col_idx = vec![0u32; total_nnz];
    let mut sign_bits = vec![0u8; total_nnz]; // 0 = add, 1 = subtract (GLV sign only)
    let mut write_pos = row_ptr.clone();

    for k in 0..k_eff {
        let n = scalars[k].len();
        for col in col_start..col_end {
            let (scalar_idx, is_phi) = if use_glv {
                if col < n {
                    (col, false)
                } else {
                    (col - n, true)
                }
            } else {
                (col, false)
            };

            if scalar_idx >= n {
                continue;
            }

            if use_glv {
                let (coeffs, signs) = decompose_scalar_2d(scalars[k][scalar_idx]);
                let (mag, sign_pos) = if is_phi {
                    (&coeffs[1], signs[1])
                } else {
                    (&coeffs[0], signs[0])
                };
                let u = extract_digit_u(mag, window_idx, window_bits) as usize;
                if u != 0 {
                    let bucket = u - 1;
                    let pos = write_pos[k][bucket];
                    col_idx[pos] = col as u32;
                    sign_bits[pos] = (!sign_pos) as u8; // if negative half → subtract point
                    write_pos[k][bucket] += 1;
                }
            } else {
                let mag = &scalars[k][scalar_idx].into_bigint();
                let u = extract_digit_u(mag, window_idx, window_bits) as usize;
                if u != 0 {
                    let bucket = u - 1;
                    let pos = write_pos[k][bucket];
                    col_idx[pos] = col as u32;
                    sign_bits[pos] = 0; // positive by definition
                    write_pos[k][bucket] += 1;
                }
            }
        }
    }

    // Debug assertions for CSR consistency
    #[cfg(debug_assertions)]
    {
        for k in 0..k_eff {
            // Row monotonicity
            for b in 0..b_max {
                debug_assert!(
                    row_ptr[k][b] <= row_ptr[k][b + 1],
                    "Row pointers not monotonic: k={}, b={}, row_ptr[{}]={} > row_ptr[{}]={}",
                    k,
                    b,
                    b,
                    row_ptr[k][b],
                    b + 1,
                    row_ptr[k][b + 1]
                );
            }

            // NNZ accounting
            let start_offset = if k == 0 { 0 } else { row_ptr[k - 1][b_max] };
            let end_offset = row_ptr[k][b_max];
            let total_nnz_k = end_offset - start_offset;
            let sum_counts: usize = counts[k].iter().sum();
            debug_assert_eq!(
                total_nnz_k, sum_counts,
                "NNZ mismatch for k={}: total_nnz={} != sum_counts={}",
                k, total_nnz_k, sum_counts
            );
        }

        // Bounds checking
        for i in 0..col_idx.len() {
            debug_assert!(
                col_idx[i] >= col_start as u32 && col_idx[i] < col_end as u32,
                "col_idx[{}]={} out of bounds [{}, {})",
                i,
                col_idx[i],
                col_start,
                col_end
            );
            debug_assert!(
                sign_bits[i] <= 1,
                "sign_bits[{}]={} not in {{0,1}}",
                i,
                sign_bits[i]
            );
        }
    }

    CsrBuckets {
        row_ptr,
        col_idx,
        sign_bits,
        b_hi,
    }
}

/// Perform segmented reduction for buckets using CSR
fn segmented_bucket_reduction(
    csr: &CsrBuckets,
    bases: &[G1Affine],
    phi_bases: &[G1Affine],
    use_glv: bool,
) -> Vec<Vec<G1Projective>> {
    let k_eff = csr.row_ptr.len();
    let b_max = csr.row_ptr[0].len() - 1;
    let n = bases.len();

    // Parallel reduction over (k, b) pairs
    let buckets: Vec<Vec<G1Projective>> = (0..k_eff)
        .into_par_iter()
        .map(|k| {
            let mut bucket_vec = vec![G1Projective::zero(); b_max];

            for b in 0..b_max {
                let start = csr.row_ptr[k][b];
                let end = csr.row_ptr[k][b + 1];

                let mut acc = G1Projective::zero();

                for i in start..end {
                    let col = csr.col_idx[i] as usize;
                    let sign = csr.sign_bits[i];

                    let point = if use_glv && col >= n {
                        phi_bases[col - n]
                    } else {
                        bases[col]
                    };

                    if sign == 0 {
                        acc += point;
                    } else {
                        acc -= point;
                    }
                }

                bucket_vec[b] = acc;
            }

            bucket_vec
        })
        .collect();

    buckets
}

/// Add k*p to dst using binary ladder (for small k)
#[inline]
fn add_mul_small(dst: &mut G1Projective, p: &G1Projective, mut k: usize) {
    if k == 0 {
        return;
    }
    let mut base = *p;
    while k > 0 {
        if k & 1 == 1 {
            *dst += base;
        }
        base.double_in_place();
        k >>= 1;
    }
}

/// Compute running sum and fold into accumulator (gap-skipping for sparse buckets)
fn running_sum_and_fold(
    buckets: &[Vec<G1Projective>],
    _b_hi: &[usize],
    accumulators: &mut [G1Projective],
    window_bits: usize,
) {
    for (k, row) in buckets.iter().enumerate() {
        // Collect non-empty bucket indices
        let mut nz: Vec<usize> = Vec::with_capacity(row.len().min(256));
        for (i, bsum) in row.iter().enumerate() {
            if !bsum.is_zero() {
                nz.push(i + 1); // Store as 1-based bucket index
            }
        }

        let acc = &mut accumulators[k];

        if nz.is_empty() {
            // Just do 2^w doublings; nothing to add
            for _ in 0..window_bits {
                acc.double_in_place();
            }
            continue;
        }

        nz.sort_unstable(); // Ascending order
        let mut running_sum = G1Projective::zero();
        let mut window_sum = G1Projective::zero();
        let mut prev = row.len() + 1; // Start "above" max bucket

        // Walk from highest bucket down, skipping gaps
        for &b in nz.iter().rev() {
            let gap = prev - b - 1; // Number of empty buckets above b
            if gap > 0 {
                add_mul_small(&mut window_sum, &running_sum, gap);
            }
            running_sum += &row[b - 1]; // Add this bucket
            window_sum += &running_sum; // Account for b itself
            prev = b;
        }

        // Final tail below smallest non-empty bucket
        if prev > 1 {
            add_mul_small(&mut window_sum, &running_sum, prev - 1);
        }

        // Fold: ACC = 2^w * ACC + window_sum
        for _ in 0..window_bits {
            acc.double_in_place();
        }
        *acc += &window_sum;
    }
}

/// Compute M MSMs of size N using Tile-K (shared bases)
///
/// # Arguments
/// * `bases` - Shared base points, length N (affine)
/// * `scalars` - M slices, each length N (scalars in Fr)
/// * `cfg` - Tuning parameters
///
/// # Returns
/// Vec<G1Affine> of length M: one MSM result per scalar slice
pub fn msm_batched_bn254_tile_k(
    bases: &[G1Affine],
    scalars: &[&[Fr]],
    cfg: &BatchedMsmConfig,
) -> Vec<G1Affine> {
    use std::time::Instant;

    let start_total = Instant::now();
    let m = scalars.len();
    let n = bases.len();

    if m == 0 || n == 0 {
        return vec![];
    }

    // Validate inputs
    for (i, s) in scalars.iter().enumerate() {
        assert_eq!(
            s.len(),
            n,
            "scalars[{}].len() = {} != bases.len() = {}",
            i,
            s.len(),
            n
        );
    }

    let window_bits = cfg.window_bits.unwrap_or_else(|| choose_window_size(n));
    let tile_k = cfg.tile_k;
    let col_chunk = cfg.col_chunk;
    let use_glv = cfg.use_glv;

    let num_windows = (Fr::MODULUS_BIT_SIZE as usize + window_bits - 1) / window_bits;

    // Precompute phi(bases) if using GLV
    let start_phi = Instant::now();
    let phi_bases: Vec<G1Affine> = if use_glv {
        bases
            .par_iter()
            .map(|b| glv_endomorphism(&b.into_group()).into_affine())
            .collect()
    } else {
        vec![]
    };
    println!("[TIMING] phi_bases precompute: {:?}", start_phi.elapsed());

    let logical_cols = if use_glv { 2 * n } else { n };

    let mut results = vec![G1Projective::zero(); m];

    // Process in tiles of K MSMs
    let start_tiles = Instant::now();
    let mut total_glv_time = std::time::Duration::ZERO;
    let mut total_digits_time = std::time::Duration::ZERO;
    let mut total_csr_time = std::time::Duration::ZERO;
    let mut total_reduction_time = std::time::Duration::ZERO;
    let mut total_accumulation_time = std::time::Duration::ZERO;
    let mut total_fold_time = std::time::Duration::ZERO;

    for tile_start in (0..m).step_by(tile_k) {
        let tile_end = (tile_start + tile_k).min(m);
        let k_eff = tile_end - tile_start;

        let scalars_tile: Vec<&[Fr]> = scalars[tile_start..tile_end].to_vec();

        // Precompute GLV decompositions for this tile
        let start_glv = Instant::now();
        let glv_splits: Vec<GlvSplit> = scalars_tile
            .par_iter()
            .map(|scalars_k| precompute_glv_tile(scalars_k, use_glv))
            .collect();
        total_glv_time += start_glv.elapsed();

        // Precompute ALL window digits ONCE for this tile
        let start_digits = Instant::now();
        let all_window_digits: Vec<AllWindowDigits> = glv_splits
            .par_iter()
            .map(|glv| compute_all_window_digits(glv, num_windows, window_bits, use_glv))
            .collect();
        total_digits_time += start_digits.elapsed();

        let all_digits_refs: Vec<&AllWindowDigits> = all_window_digits.iter().collect();
        let glv_splits_refs: Vec<&GlvSplit> = glv_splits.iter().collect();

        let mut accumulators = vec![G1Projective::zero(); k_eff];
        let b_max = (1usize << window_bits) - 1;
        let mut window_buckets = vec![vec![G1Projective::zero(); b_max]; k_eff];

        // Process windows from high to low
        for window_idx in (0..num_windows).rev() {
            // Zero out window buckets for reuse
            for k in 0..k_eff {
                for b in 0..b_max {
                    window_buckets[k][b] = G1Projective::zero();
                }
            }

            // Track highest used bucket across all chunks for this window
            let mut b_hi_window = vec![0usize; k_eff];

            // Process columns in chunks and accumulate into buckets
            for col_start in (0..logical_cols).step_by(col_chunk) {
                let col_end = (col_start + col_chunk).min(logical_cols);

                // Build CSR using precomputed digits
                let start_csr = Instant::now();
                let csr = build_csr_from_digits(
                    &all_digits_refs,
                    &glv_splits_refs,
                    window_idx,
                    col_start,
                    col_end,
                    use_glv,
                    b_max,
                );
                total_csr_time += start_csr.elapsed();

                // Update b_hi_window with max from this chunk
                for k in 0..k_eff {
                    b_hi_window[k] = b_hi_window[k].max(csr.b_hi[k]);
                }

                // Perform bucket reduction
                let start_reduction = Instant::now();
                let chunk_buckets = segmented_bucket_reduction(&csr, bases, &phi_bases, use_glv);
                total_reduction_time += start_reduction.elapsed();

                // Accumulate into window buckets
                let start_accum = Instant::now();
                for k in 0..k_eff {
                    for b in 0..b_max {
                        window_buckets[k][b] += &chunk_buckets[k][b];
                    }
                }
                total_accumulation_time += start_accum.elapsed();
            }

            // Now fold this window's buckets into accumulators
            let start_fold = Instant::now();
            running_sum_and_fold(
                &window_buckets,
                &b_hi_window,
                &mut accumulators,
                window_bits,
            );
            total_fold_time += start_fold.elapsed();
        }

        // Write results for this tile
        for (i, acc) in accumulators.iter().enumerate() {
            results[tile_start + i] = *acc;
        }
    }
    println!("[TIMING] tiles processing: {:?}", start_tiles.elapsed());
    println!("  └─ GLV decomposition: {:?}", total_glv_time);
    println!("  └─ Window digits: {:?}", total_digits_time);
    println!("  └─ CSR build: {:?}", total_csr_time);
    println!("  └─ Bucket reduction: {:?}", total_reduction_time);
    println!("  └─ Bucket accumulation: {:?}", total_accumulation_time);
    println!("  └─ Fold: {:?}", total_fold_time);

    // Batch normalize to affine
    let start_normalize = Instant::now();
    let result = results.into_iter().map(|p| p.into_affine()).collect();
    println!("[TIMING] batch normalize: {:?}", start_normalize.elapsed());
    println!("[TIMING] TOTAL: {:?}", start_total.elapsed());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::VariableBaseMSM;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_extract_digit() {
        // 0b10110101 = 181
        let s = Fr::from(0b10110101u64).into_bigint();

        // Window size 3, extract at position 0
        let d0 = extract_digit_u(&s, 0, 3);
        assert_eq!(d0, 0b101); // bits [0,1,2] = 101 = 5

        // Window size 3, extract at position 1
        let d1 = extract_digit_u(&s, 1, 3);
        assert_eq!(d1, 0b110); // bits [3,4,5] = 011 = 6

        // Window size 4, extract at position 0
        let d2 = extract_digit_u(&s, 0, 4);
        assert_eq!(d2, 0b0101); // bits [0,1,2,3] = 0101 = 5

        // Test edge case: window beyond scalar bits
        let d3 = extract_digit_u(&s, 100, 4);
        assert_eq!(d3, 0);
    }

    #[test]
    fn test_extract_unsigned_digit() {
        // Test unsigned digit extraction
        let s = Fr::from(0b1111u64).into_bigint(); // 15 in 4 bits

        // Window size 4: should extract 15
        let d0 = extract_digit_u(&s, 0, 4);
        assert_eq!(d0, 15);

        // Window size 4 on value 7
        let s2 = Fr::from(0b0111u64).into_bigint();
        let d1 = extract_digit_u(&s2, 0, 4);
        assert_eq!(d1, 7);

        // Window size 4 on value 8
        let s3 = Fr::from(0b1000u64).into_bigint();
        let d2 = extract_digit_u(&s3, 0, 4);
        assert_eq!(d2, 8);
    }

    #[test]
    fn test_running_sum_and_fold() {
        use ark_ec::CurveGroup;

        // Create simple bucket structure: bucket[0] = G, bucket[1] = 2G
        let g = G1Affine::generator().into_group();
        let buckets = vec![vec![g, g.double()]];
        let mut accumulators = vec![G1Projective::zero()];

        // With window_bits=4, should compute:
        // running_sum after bucket[1]: 2G
        // window_sum after bucket[1]: 2G
        // running_sum after bucket[0]: 2G + G = 3G
        // window_sum after bucket[0]: 2G + 3G = 5G
        // acc = 2^4 * 0 + 5G = 5G

        let b_hi = vec![2]; // bucket 0 and 1 are used
        running_sum_and_fold(&buckets, &b_hi, &mut accumulators, 4);

        let expected = g * Fr::from(5u64);
        assert_eq!(accumulators[0], expected, "Running sum fold incorrect");
    }

    #[test]
    fn test_segmented_bucket_reduction() {
        // Test with 2 bases, simple CSR structure
        let bases = vec![
            G1Affine::generator(),
            G1Affine::generator().into_group().double().into_affine(),
        ];
        let phi_bases = vec![];

        // Create CSR: MSM 0 has bucket 0 with base 0 (positive) and base 1 (negative)
        let csr = CsrBuckets {
            row_ptr: vec![vec![0, 2, 2]], // 1 MSM, 2 buckets (only bucket 0 used)
            col_idx: vec![0, 1],
            sign_bits: vec![0, 1], // base 0 positive, base 1 negative
            b_hi: vec![1],
        };

        let result = segmented_bucket_reduction(&csr, &bases, &phi_bases, false);

        // bucket[0] should be G - 2G = -G
        let expected =
            G1Affine::generator().into_group() - G1Affine::generator().into_group().double();
        assert_eq!(result[0][0], expected, "Bucket reduction incorrect");
    }

    #[test]
    fn test_build_csr_for_window() {
        // Test CSR construction with simple scalar
        let bases = vec![G1Affine::generator()];
        let scalars_vec = vec![Fr::from(5u64)]; // 5 = 0b101
        let scalars = vec![scalars_vec.as_slice()];

        // For window 0, w=4: digit = 5, bucket = 4
        let csr = build_csr_for_window(&scalars, 0, 4, 0, 1, false);

        assert_eq!(csr.row_ptr[0][0], 0);
        assert_eq!(csr.row_ptr[0][5], 1); // bucket 4 has 1 entry
        assert_eq!(csr.col_idx[0], 0);
        assert_eq!(csr.sign_bits[0], 0); // positive
    }

    #[test]
    fn test_batched_msm_tiny() {
        // Simplest possible test: 1 MSM, 2 bases
        let bases = vec![
            G1Affine::generator(),
            G1Affine::generator().into_group().double().into_affine(),
        ];
        let scalars = vec![Fr::from(3u64), Fr::from(5u64)];
        let scalars_ref: Vec<&[Fr]> = vec![&scalars];

        let mut cfg = BatchedMsmConfig::default();
        cfg.use_glv = false;
        cfg.tile_k = 1;

        let results = msm_batched_bn254_tile_k(&bases, &scalars_ref, &cfg);

        // Expected: 3*G + 5*(2G) = 3*G + 10*G = 13*G
        let expected = G1Affine::generator().into_group() * Fr::from(13u64);
        assert_eq!(results[0], expected.into_affine());
    }

    #[test]
    fn test_batched_msm_power_of_two() {
        // Test with 256 = 2^8, should be exactly 8 doublings
        let bases = vec![G1Affine::generator()];
        let scalars = vec![Fr::from(256u64)];
        let scalars_ref: Vec<&[Fr]> = vec![&scalars];

        let mut cfg = BatchedMsmConfig::default();
        cfg.use_glv = false;
        cfg.tile_k = 1;
        cfg.window_bits = Some(4);

        let results = msm_batched_bn254_tile_k(&bases, &scalars_ref, &cfg);

        let expected = G1Affine::generator().into_group() * Fr::from(256u64);
        assert_eq!(results[0], expected.into_affine(), "Power of 2 MSM failed");
    }

    #[test]
    fn test_batched_msm_two_bits() {
        // Test with 17 = 0b10001, has 2 non-zero bits
        let bases = vec![G1Affine::generator()];
        let scalars = vec![Fr::from(17u64)];
        let scalars_ref: Vec<&[Fr]> = vec![&scalars];

        let mut cfg = BatchedMsmConfig::default();
        cfg.use_glv = false;
        cfg.tile_k = 1;
        cfg.window_bits = Some(4);

        let results = msm_batched_bn254_tile_k(&bases, &scalars_ref, &cfg);

        let expected = G1Affine::generator().into_group() * Fr::from(17u64);
        assert_eq!(results[0], expected.into_affine(), "Two bits MSM failed");
    }

    #[test]
    fn test_batched_msm_15() {
        // Test with 15 = 0b1111 in window 0 -> signed digit -1
        // Should compute -1 * G = -G
        let bases = vec![G1Affine::generator()];
        let scalars = vec![Fr::from(15u64)];
        let scalars_ref: Vec<&[Fr]> = vec![&scalars];

        let mut cfg = BatchedMsmConfig::default();
        cfg.use_glv = false;
        cfg.tile_k = 1;
        cfg.window_bits = Some(4);

        let results = msm_batched_bn254_tile_k(&bases, &scalars_ref, &cfg);

        let expected = G1Affine::generator().into_group() * Fr::from(15u64);
        assert_eq!(
            results[0],
            expected.into_affine(),
            "Signed digit MSM failed"
        );
    }

    #[test]
    fn test_batched_msm_larger_scalar() {
        // Test with a larger scalar that requires multiple windows
        let bases = vec![G1Affine::generator()];
        let scalars = vec![Fr::from(123456u64)];
        let scalars_ref: Vec<&[Fr]> = vec![&scalars];

        let mut cfg = BatchedMsmConfig::default();
        cfg.use_glv = false;
        cfg.tile_k = 1;
        cfg.window_bits = Some(4); // Use smaller window for easier debugging

        let results = msm_batched_bn254_tile_k(&bases, &scalars_ref, &cfg);

        let expected = G1Affine::generator().into_group() * Fr::from(123456u64);
        assert_eq!(
            results[0],
            expected.into_affine(),
            "Multi-window MSM failed"
        );
    }

    #[test]
    fn test_batched_msm_small() {
        let mut rng = test_rng();

        let n = 16;
        let m = 4;

        let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();

        let scalars_owned: Vec<Vec<Fr>> = (0..m)
            .map(|_| (0..n).map(|_| Fr::rand(&mut rng)).collect())
            .collect();

        let scalars: Vec<&[Fr]> = scalars_owned.iter().map(|v| v.as_slice()).collect();

        let cfg = BatchedMsmConfig::default();
        let results = msm_batched_bn254_tile_k(&bases, &scalars, &cfg);

        assert_eq!(results.len(), m);

        // Verify against naive MSM
        for i in 0..m {
            let expected = G1Projective::msm(&bases, &scalars_owned[i]).unwrap();
            assert_eq!(results[i], expected.into_affine(), "MSM {} mismatch", i);
        }
    }

    #[test]
    fn test_batched_msm_no_glv() {
        let mut rng = test_rng();

        let n = 4;
        let m = 1;

        let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();

        let scalars_owned: Vec<Vec<Fr>> = (0..m)
            .map(|_| (0..n).map(|_| Fr::rand(&mut rng)).collect())
            .collect();

        let scalars: Vec<&[Fr]> = scalars_owned.iter().map(|v| v.as_slice()).collect();

        let mut cfg = BatchedMsmConfig::default();
        cfg.use_glv = false;

        let results = msm_batched_bn254_tile_k(&bases, &scalars, &cfg);

        for i in 0..m {
            let expected = G1Projective::msm(&bases, &scalars_owned[i]).unwrap();
            assert_eq!(results[i], expected.into_affine(), "MSM {} mismatch", i);
        }
    }
}
