use super::{Fp, FpConfig};
use crate::{
    biginteger::arithmetic as fa, BigInt, BigInteger, PrimeField, SqrtPrecomputation, Zero,
};
use ark_ff_macros::unroll_for_loops;
use ark_std::marker::PhantomData;

pub const PRECOMP_TABLE_SIZE: usize = 1 << 14;

/// A trait that specifies the constants and arithmetic procedures
/// for Montgomery arithmetic over the prime field defined by `MODULUS`.
///
/// # Note
/// Manual implementation of this trait is not recommended unless one wishes
/// to specialize arithmetic methods. Instead, the
/// [`MontConfig`][`ark_ff_macros::MontConfig`] derive macro should be used.
pub trait MontConfig<const N: usize>: 'static + Sync + Send + Sized {
    /// The modulus of the field.
    const MODULUS: BigInt<N>;

    /// Let `M` be the power of 2^64 nearest to `Self::MODULUS_BITS`. Then
    /// `R = M % Self::MODULUS`.
    const R: BigInt<N> = Self::MODULUS.montgomery_r();

    /// R2 = R^2 % Self::MODULUS
    const R2: BigInt<N> = Self::MODULUS.montgomery_r2();

    /// INV = -MODULUS^{-1} mod 2^64
    const INV: u64 = inv::<Self, N>();

    /// A multiplicative generator of the field.
    /// `Self::GENERATOR` is an element having multiplicative order
    /// `Self::MODULUS - 1`.
    const GENERATOR: Fp<MontBackend<Self, N>, N>;

    /// Can we use the no-carry optimization for multiplication
    /// outlined [here](https://hackmd.io/@gnark/modular_multiplication)?
    ///
    /// This optimization applies if
    /// (a) `Self::MODULUS[N-1] < u64::MAX >> 1`, and
    /// (b) the bits of the modulus are not all 1.
    #[doc(hidden)]
    const CAN_USE_NO_CARRY_MUL_OPT: bool = can_use_no_carry_mul_optimization::<Self, N>();

    /// Can we use the no-carry optimization for squaring
    /// outlined [here](https://hackmd.io/@gnark/modular_multiplication)?
    ///
    /// This optimization applies if
    /// (a) `Self::MODULUS[N-1] < u64::MAX >> 2`, and
    /// (b) the bits of the modulus are not all 1.
    #[doc(hidden)]
    const CAN_USE_NO_CARRY_SQUARE_OPT: bool = can_use_no_carry_mul_optimization::<Self, N>();

    /// Does the modulus have a spare unused bit
    ///
    /// This condition applies if
    /// (a) `Self::MODULUS[N-1] >> 63 == 0`
    #[doc(hidden)]
    const MODULUS_HAS_SPARE_BIT: bool = modulus_has_spare_bit::<Self, N>();

    /// 2^s root of unity computed by GENERATOR^t
    const TWO_ADIC_ROOT_OF_UNITY: Fp<MontBackend<Self, N>, N>;

    /// An integer `b` such that there exists a multiplicative subgroup
    /// of size `b^k` for some integer `k`.
    const SMALL_SUBGROUP_BASE: Option<u32> = None;

    /// The integer `k` such that there exists a multiplicative subgroup
    /// of size `Self::SMALL_SUBGROUP_BASE^k`.
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = None;

    /// GENERATOR^((MODULUS-1) / (2^s *
    /// SMALL_SUBGROUP_BASE^SMALL_SUBGROUP_BASE_ADICITY)).
    /// Used for mixed-radix FFT.
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<Fp<MontBackend<Self, N>, N>> = None;

    /// Precomputed material for use when computing square roots.
    /// The default is to use the standard Tonelli-Shanks algorithm.
    const SQRT_PRECOMP: Option<SqrtPrecomputation<Fp<MontBackend<Self, N>, N>>> =
        sqrt_precomputation::<N, Self>();

    #[allow(long_running_const_eval)]
    const SMALL_ELEMENT_MONTGOMERY_PRECOMP: [Fp<MontBackend<Self, N>, N>; PRECOMP_TABLE_SIZE] =
        small_element_montgomery_precomputation::<N, Self>();

    /// (MODULUS + 1) / 4 when MODULUS % 4 == 3. Used for square root precomputations.
    #[doc(hidden)]
    const MODULUS_PLUS_ONE_DIV_FOUR: Option<BigInt<N>> = {
        match Self::MODULUS.mod_4() == 3 {
            true => {
                let (modulus_plus_one, carry) =
                    Self::MODULUS.const_add_with_carry(&BigInt::<N>::one());
                let mut result = modulus_plus_one.divide_by_2_round_down();
                // Since modulus_plus_one is even, dividing by 2 results in a MSB of 0.
                // Thus we can set MSB to `carry` to get the correct result of (MODULUS + 1) // 2:
                result.0[N - 1] |= (carry as u64) << 63;
                Some(result.divide_by_2_round_down())
            },
            false => None,
        }
    };

    /// Number of spare bits (i.e. significant bits equal to 0) in the modulus `p`
    #[doc(hidden)]
    const MODULUS_NUM_SPARE_BITS: u32 = Self::MODULUS.num_spare_bits();

    /// Represents the modulus `p` as an N+1 limb number `(low_n_limbs, high_limb)` with the highest limb being 0.
    #[doc(hidden)]
    const MODULUS_NPLUS1: ([u64; N], u64) = {
        // The highest limb (hi[N-1]) remains 0.
        (Self::MODULUS.0, 0u64)
    };

    /// Represents `2*p` as an N+1 limb number `(low_n_limbs, high_limb)`.
    #[doc(hidden)]
    const MODULUS_TIMES_2_NPLUS1: ([u64; N], u64) = {
        let (mod2_arr, mod2_carry) = Self::MODULUS.const_mul2_with_carry();
        // The highest limb is the carry bit
        (mod2_arr.0, mod2_carry as u64)
    };

    /// Represents `3*p` as an N+1 limb number `(low_n_limbs, high_limb)`.
    #[doc(hidden)]
    const MODULUS_TIMES_3_NPLUS1: ([u64; N], u64) = {
        // Add MODULUS_NPLUS1 and MODULUS_TIMES_2_NPLUS1 using the new format.
        let (a_lo, a_hi) = Self::MODULUS_NPLUS1;
        let (b_lo, b_hi) = Self::MODULUS_TIMES_2_NPLUS1;
        let mut sum_lo = [0u64; N];
        let mut carry = 0u64;

        // Const addition loop
        let mut i = 0;
        while i < N {
            let tmp = (a_lo[i] as u128) + (b_lo[i] as u128) + (carry as u128);
            sum_lo[i] = tmp as u64;
            carry = (tmp >> 64) as u64;
            i += 1;
        }

        let sum_hi = a_hi + b_hi + carry; // Add high limbs and final carry
        (sum_lo, sum_hi)
    };

    /// Barrett reduction constant: $R' = 2^{\text{MODULUS_BITS}}$.
    #[doc(hidden)]
    const BARRETT_RPRIME: (BigInt<N>, bool) = {
        let num_spare_bits = Self::MODULUS.num_spare_bits();
        assert!(num_spare_bits <= 64);
        if num_spare_bits == 0 {
            (BigInt::<N>::zero(), true)
        } else {
            (BigInt::<N>::pow_2(64 - num_spare_bits), false)
        }
    };

    /// Barrett reduction constant: mu = floor( r * R' / (2 * MODULUS) )
    /// = floor( 2^(64 * (N + 1) - num_spare_bits(MODULUS) - 1) / MODULUS )
    #[doc(hidden)]
    const BARRETT_MU: u64 = {
        // Compute Barrett mu = floor(2^(modulus_bits + 63) / modulus)
        assert!(Self::MODULUS.num_spare_bits() < 64);
        let r_times_r_prime_over_2 = crate::const_helpers::RBuffer::<N>(
            [0u64; N],
            1 << (63 - Self::MODULUS.num_spare_bits()),
        );
        // Use const_quotient! to compute the quotient
        let result: BigInt<N> = const_quotient!(r_times_r_prime_over_2, &Self::MODULUS);
        // Result should be a u64
        result.0[0]
    };

    /// Sets `a = a + b`.
    #[inline(always)]
    fn add_assign(a: &mut Fp<MontBackend<Self, N>, N>, b: &Fp<MontBackend<Self, N>, N>) {
        // This cannot exceed the backing capacity.
        let c = a.0.add_with_carry(&b.0);
        // However, it may need to be reduced
        if Self::MODULUS_HAS_SPARE_BIT {
            a.subtract_modulus()
        } else {
            a.subtract_modulus_with_carry(c)
        }
    }

    /// Sets `a = a - b`.
    #[inline(always)]
    fn sub_assign(a: &mut Fp<MontBackend<Self, N>, N>, b: &Fp<MontBackend<Self, N>, N>) {
        // If `other` is larger than `self`, add the modulus to self first.
        if b.0 > a.0 {
            a.0.add_with_carry(&Self::MODULUS);
        }
        a.0.sub_with_borrow(&b.0);
    }

    /// Sets `a = 2 * a`.
    #[inline(always)]
    fn double_in_place(a: &mut Fp<MontBackend<Self, N>, N>) {
        // This cannot exceed the backing capacity.
        let c = a.0.mul2();
        // However, it may need to be reduced.
        if Self::MODULUS_HAS_SPARE_BIT {
            a.subtract_modulus()
        } else {
            a.subtract_modulus_with_carry(c)
        }
    }

    /// Sets `a = -a`.
    #[inline(always)]
    fn neg_in_place(a: &mut Fp<MontBackend<Self, N>, N>) {
        if !a.is_zero() {
            let mut tmp = Self::MODULUS;
            tmp.sub_with_borrow(&a.0);
            a.0 = tmp;
        }
    }

    /// This modular multiplication algorithm uses Montgomery
    /// reduction for efficient implementation. It also additionally
    /// uses the "no-carry optimization" outlined
    /// [here](https://hackmd.io/@gnark/modular_multiplication) if
    /// `Self::MODULUS` has (a) a non-zero MSB, and (b) at least one
    /// zero bit in the rest of the modulus.
    #[unroll_for_loops(12)]
    #[inline(always)]
    fn mul_assign(a: &mut Fp<MontBackend<Self, N>, N>, b: &Fp<MontBackend<Self, N>, N>) {
        // No-carry optimisation applied to CIOS
        if Self::CAN_USE_NO_CARRY_MUL_OPT {
            if N <= 6
                && N > 1
                && cfg!(all(
                    feature = "asm",
                    target_feature = "bmi2",
                    target_feature = "adx",
                    target_arch = "x86_64"
                ))
            {
                #[cfg(
                    all(
                        feature = "asm",
                        target_feature = "bmi2",
                        target_feature = "adx",
                        target_arch = "x86_64"
                    )
                )]
                #[allow(unsafe_code, unused_mut)]
                #[rustfmt::skip]

                // Tentatively avoid using assembly for `N == 1`.
                match N {
                    2 => { ark_ff_asm::x86_64_asm_mul!(2, (a.0).0, (b.0).0); },
                    3 => { ark_ff_asm::x86_64_asm_mul!(3, (a.0).0, (b.0).0); },
                    4 => { ark_ff_asm::x86_64_asm_mul!(4, (a.0).0, (b.0).0); },
                    5 => { ark_ff_asm::x86_64_asm_mul!(5, (a.0).0, (b.0).0); },
                    6 => { ark_ff_asm::x86_64_asm_mul!(6, (a.0).0, (b.0).0); },
                    _ => unsafe { ark_std::hint::unreachable_unchecked() },
                };
            } else {
                let mut r = [0u64; N];

                for i in 0..N {
                    let mut carry1 = 0u64;
                    r[0] = fa::mac(r[0], (a.0).0[0], (b.0).0[i], &mut carry1);

                    let k = r[0].wrapping_mul(Self::INV);

                    let mut carry2 = 0u64;
                    fa::mac_discard(r[0], k, Self::MODULUS.0[0], &mut carry2);

                    for j in 1..N {
                        r[j] = fa::mac_with_carry(r[j], (a.0).0[j], (b.0).0[i], &mut carry1);
                        r[j - 1] = fa::mac_with_carry(r[j], k, Self::MODULUS.0[j], &mut carry2);
                    }
                    r[N - 1] = carry1 + carry2;
                }
                (a.0).0.copy_from_slice(&r);
            }
            a.subtract_modulus();
        } else {
            // Alternative implementation
            // Implements CIOS.
            let (carry, res) = a.mul_without_cond_subtract(b);
            *a = res;

            if Self::MODULUS_HAS_SPARE_BIT {
                a.subtract_modulus_with_carry(carry);
            } else {
                a.subtract_modulus();
            }
        }
    }

    #[inline(always)]
    #[unroll_for_loops(12)]
    fn square_in_place(a: &mut Fp<MontBackend<Self, N>, N>) {
        if N == 1 {
            // We default to multiplying with `a` using the `Mul` impl
            // for the N == 1 case
            *a *= *a;
            return;
        }
        if Self::CAN_USE_NO_CARRY_SQUARE_OPT
            && (2..=6).contains(&N)
            && cfg!(all(
                feature = "asm",
                target_feature = "bmi2",
                target_feature = "adx",
                target_arch = "x86_64"
            ))
        {
            #[cfg(all(
                feature = "asm",
                target_feature = "bmi2",
                target_feature = "adx",
                target_arch = "x86_64"
            ))]
            #[allow(unsafe_code, unused_mut)]
            #[rustfmt::skip]
            match N {
                2 => { ark_ff_asm::x86_64_asm_square!(2, (a.0).0); },
                3 => { ark_ff_asm::x86_64_asm_square!(3, (a.0).0); },
                4 => { ark_ff_asm::x86_64_asm_square!(4, (a.0).0); },
                5 => { ark_ff_asm::x86_64_asm_square!(5, (a.0).0); },
                6 => { ark_ff_asm::x86_64_asm_square!(6, (a.0).0); },
                _ => unsafe { ark_std::hint::unreachable_unchecked() },
            };
            a.subtract_modulus();
            return;
        }

        let mut r = crate::const_helpers::MulBuffer::<N>::zeroed();

        let mut carry = 0;
        for i in 0..(N - 1) {
            for j in (i + 1)..N {
                r[i + j] = fa::mac_with_carry(r[i + j], (a.0).0[i], (a.0).0[j], &mut carry);
            }
            r.b1[i] = carry;
            carry = 0;
        }

        r.b1[N - 1] = r.b1[N - 2] >> 63;
        for i in 2..(2 * N - 1) {
            r[2 * N - i] = (r[2 * N - i] << 1) | (r[2 * N - (i + 1)] >> 63);
        }
        r.b0[1] <<= 1;

        for i in 0..N {
            r[2 * i] = fa::mac_with_carry(r[2 * i], (a.0).0[i], (a.0).0[i], &mut carry);
            carry = fa::adc(&mut r[2 * i + 1], 0, carry);
        }
        // Montgomery reduction
        let mut carry2 = 0;
        for i in 0..N {
            let k = r[i].wrapping_mul(Self::INV);
            carry = 0;
            fa::mac_discard(r[i], k, Self::MODULUS.0[0], &mut carry);
            for j in 1..N {
                r[j + i] = fa::mac_with_carry(r[j + i], k, Self::MODULUS.0[j], &mut carry);
            }
            carry2 = fa::adc(&mut r.b1[i], carry, carry2);
        }
        (a.0).0.copy_from_slice(&r.b1);
        if Self::MODULUS_HAS_SPARE_BIT {
            a.subtract_modulus();
        } else {
            a.subtract_modulus_with_carry(carry2 != 0);
        }
    }

    fn inverse(a: &Fp<MontBackend<Self, N>, N>) -> Option<Fp<MontBackend<Self, N>, N>> {
        if a.is_zero() {
            return None;
        }
        // Guajardo Kumar Paar Pelzl
        // Efficient Software-Implementation of Finite Fields with Applications to
        // Cryptography
        // Algorithm 16 (BEA for Inversion in Fp)

        let one = BigInt::from(1u64);

        let mut u = a.0;
        let mut v = Self::MODULUS;
        let mut b = Fp::new_unchecked(Self::R2); // Avoids unnecessary reduction step.
        let mut c = Fp::zero();

        while u != one && v != one {
            while u.is_even() {
                u.div2();

                if b.0.is_even() {
                    b.0.div2();
                } else {
                    let carry = b.0.add_with_carry(&Self::MODULUS);
                    b.0.div2();
                    if !Self::MODULUS_HAS_SPARE_BIT && carry {
                        (b.0).0[N - 1] |= 1 << 63;
                    }
                }
            }

            while v.is_even() {
                v.div2();

                if c.0.is_even() {
                    c.0.div2();
                } else {
                    let carry = c.0.add_with_carry(&Self::MODULUS);
                    c.0.div2();
                    if !Self::MODULUS_HAS_SPARE_BIT && carry {
                        (c.0).0[N - 1] |= 1 << 63;
                    }
                }
            }

            if v < u {
                u.sub_with_borrow(&v);
                b -= &c;
            } else {
                v.sub_with_borrow(&u);
                c -= &b;
            }
        }

        if u == one {
            Some(b)
        } else {
            Some(c)
        }
    }

    fn from_i128(r: i128) -> Option<Fp<MontBackend<Self, N>, N>> {
        // TODO: small table for signed values?
        Some(Fp::new_unchecked(Self::R).mul_i128(r))
    }

    fn from_u128(r: u128) -> Option<Fp<MontBackend<Self, N>, N>> {
        if r < PRECOMP_TABLE_SIZE as u128 {
            Some(Self::SMALL_ELEMENT_MONTGOMERY_PRECOMP[r as usize])
        } else {
            // Multiply R (one in Montgomery form) with the u128
            Some(Fp::new_unchecked(Self::R).mul_u128(r))
        }
    }

    fn from_i64(r: i64) -> Option<Fp<MontBackend<Self, N>, N>> {
        // TODO: small table for signed values?
        Some(Fp::new_unchecked(Self::R).mul_i64(r))
    }

    fn from_u64(r: u64) -> Option<Fp<MontBackend<Self, N>, N>> {
        if r < PRECOMP_TABLE_SIZE as u64 {
            Some(Self::SMALL_ELEMENT_MONTGOMERY_PRECOMP[r as usize])
        } else {
            // Multiply R (one in Montgomery form) with the u64
            Some(Fp::new_unchecked(Self::R).mul_u64(r))
        }
    }

    fn from_bigint(r: BigInt<N>) -> Option<Fp<MontBackend<Self, N>, N>> {
        let mut r = Fp::new_unchecked(r);
        if r.is_zero() {
            Some(r)
        } else if r.is_geq_modulus() {
            None
        } else {
            r *= &Fp::new_unchecked(Self::R2);
            Some(r)
        }
    }

    #[inline]
    #[cfg_attr(not(target_family = "wasm"), unroll_for_loops(12))]
    #[cfg_attr(target_family = "wasm", unroll_for_loops(6))]
    #[allow(clippy::modulo_one)]
    fn into_bigint(a: Fp<MontBackend<Self, N>, N>) -> BigInt<N> {
        let mut r = (a.0).0;
        // Montgomery Reduction
        for i in 0..N {
            let k = r[i].wrapping_mul(Self::INV);
            let mut carry = 0;

            fa::mac_with_carry(r[i], k, Self::MODULUS.0[0], &mut carry);
            for j in 1..N {
                r[(j + i) % N] =
                    fa::mac_with_carry(r[(j + i) % N], k, Self::MODULUS.0[j], &mut carry);
            }
            r[i % N] = carry;
        }

        BigInt::new(r)
    }

    #[unroll_for_loops(12)]
    fn sum_of_products<const M: usize>(
        a: &[Fp<MontBackend<Self, N>, N>; M],
        b: &[Fp<MontBackend<Self, N>, N>; M],
    ) -> Fp<MontBackend<Self, N>, N> {
        // Adapted from https://github.com/zkcrypto/bls12_381/pull/84 by @str4d.

        // For a single `a x b` multiplication, operand scanning (schoolbook) takes each
        // limb of `a` in turn, and multiplies it by all of the limbs of `b` to compute
        // the result as a double-width intermediate representation, which is then fully
        // reduced at the carry. Here however we have pairs of multiplications (a_i, b_i),
        // the results of which are summed.
        //
        // The intuition for this algorithm is two-fold:
        // - We can interleave the operand scanning for each pair, by processing the jth
        //   limb of each `a_i` together. As these have the same offset within the overall
        //   operand scanning flow, their results can be summed directly.
        // - We can interleave the multiplication and reduction steps, resulting in a
        //   single bitshift by the limb size after each iteration. This means we
        //   only need to store a single extra limb overall, instead of keeping around all the
        //   intermediate results and eventually having twice as many limbs.

        let modulus_size = Self::MODULUS.const_num_bits() as usize;
        if modulus_size >= 64 * N - 1 {
            a.iter().zip(b).map(|(a, b)| *a * b).sum()
        } else if M == 2 {
            // Algorithm 2, line 2
            let result = (0..N).fold(BigInt::zero(), |mut result, j| {
                // Algorithm 2, line 3
                let mut carry_a = 0;
                let mut carry_b = 0;
                for (a, b) in a.iter().zip(b) {
                    let a = &a.0;
                    let b = &b.0;
                    let mut carry2 = 0;
                    result.0[0] = fa::mac(result.0[0], a.0[j], b.0[0], &mut carry2);
                    for k in 1..N {
                        result.0[k] = fa::mac_with_carry(result.0[k], a.0[j], b.0[k], &mut carry2);
                    }
                    carry_b = fa::adc(&mut carry_a, carry_b, carry2);
                }

                let k = result.0[0].wrapping_mul(Self::INV);
                let mut carry2 = 0;
                fa::mac_discard(result.0[0], k, Self::MODULUS.0[0], &mut carry2);
                for i in 1..N {
                    result.0[i - 1] =
                        fa::mac_with_carry(result.0[i], k, Self::MODULUS.0[i], &mut carry2);
                }
                result.0[N - 1] = fa::adc_no_carry(carry_a, carry_b, &mut carry2);
                result
            });
            let mut result = Fp::new_unchecked(result);
            result.subtract_modulus();
            debug_assert_eq!(
                a.iter().zip(b).map(|(a, b)| *a * b).sum::<Fp<_, N>>(),
                result
            );
            result
        } else {
            let chunk_size = 2 * (N * 64 - modulus_size) - 1;
            // chunk_size is at least 1, since MODULUS_BIT_SIZE is at most N * 64 - 1.
            a.chunks(chunk_size)
                .zip(b.chunks(chunk_size))
                .map(|(a, b)| {
                    // Algorithm 2, line 2
                    let result = (0..N).fold(BigInt::zero(), |mut result, j| {
                        // Algorithm 2, line 3
                        let (temp, carry) = a.iter().zip(b).fold(
                            (result, 0),
                            |(mut temp, mut carry), (Fp(a, _), Fp(b, _))| {
                                let mut carry2 = 0;
                                temp.0[0] = fa::mac(temp.0[0], a.0[j], b.0[0], &mut carry2);
                                for k in 1..N {
                                    temp.0[k] =
                                        fa::mac_with_carry(temp.0[k], a.0[j], b.0[k], &mut carry2);
                                }
                                carry = fa::adc_no_carry(carry, 0, &mut carry2);
                                (temp, carry)
                            },
                        );

                        let k = temp.0[0].wrapping_mul(Self::INV);
                        let mut carry2 = 0;
                        fa::mac_discard(temp.0[0], k, Self::MODULUS.0[0], &mut carry2);
                        for i in 1..N {
                            result.0[i - 1] =
                                fa::mac_with_carry(temp.0[i], k, Self::MODULUS.0[i], &mut carry2);
                        }
                        result.0[N - 1] = fa::adc_no_carry(carry, 0, &mut carry2);
                        result
                    });
                    let mut result = Fp::new_unchecked(result);
                    result.subtract_modulus();
                    debug_assert_eq!(
                        a.iter().zip(b).map(|(a, b)| *a * b).sum::<Fp<_, N>>(),
                        result
                    );
                    result
                })
                .sum()
        }
    }
}

/// Compute -M^{-1} mod 2^64.
pub const fn inv<T: MontConfig<N>, const N: usize>() -> u64 {
    // We compute this as follows.
    // First, MODULUS mod 2^64 is just the lower 64 bits of MODULUS.
    // Hence MODULUS mod 2^64 = MODULUS.0[0] mod 2^64.
    //
    // Next, computing the inverse mod 2^64 involves exponentiating by
    // the multiplicative group order, which is euler_totient(2^64) - 1.
    // Now, euler_totient(2^64) = 1 << 63, and so
    // euler_totient(2^64) - 1 = (1 << 63) - 1 = 1111111... (63 digits).
    // We compute this powering via standard square and multiply.
    let mut inv = 1u64;
    crate::const_for!((_i in 0..63) {
        // Square
        inv = inv.wrapping_mul(inv);
        // Multiply
        inv = inv.wrapping_mul(T::MODULUS.0[0]);
    });
    inv.wrapping_neg()
}

#[inline]
pub const fn can_use_no_carry_mul_optimization<T: MontConfig<N>, const N: usize>() -> bool {
    // Checking the modulus at compile time
    let mut all_remaining_bits_are_one = T::MODULUS.0[N - 1] == u64::MAX >> 1;
    crate::const_for!((i in 1..N) {
        all_remaining_bits_are_one  &= T::MODULUS.0[N - i - 1] == u64::MAX;
    });
    modulus_has_spare_bit::<T, N>() && !all_remaining_bits_are_one
}

#[inline]
pub const fn modulus_has_spare_bit<T: MontConfig<N>, const N: usize>() -> bool {
    T::MODULUS.0[N - 1] >> 63 == 0
}

#[inline]
pub const fn can_use_no_carry_square_optimization<T: MontConfig<N>, const N: usize>() -> bool {
    // Checking the modulus at compile time
    let top_two_bits_are_zero = T::MODULUS.0[N - 1] >> 62 == 0;
    let mut all_remaining_bits_are_one = T::MODULUS.0[N - 1] == u64::MAX >> 2;
    crate::const_for!((i in 1..N) {
        all_remaining_bits_are_one  &= T::MODULUS.0[N - i - 1] == u64::MAX;
    });
    top_two_bits_are_zero && !all_remaining_bits_are_one
}

pub const fn sqrt_precomputation<const N: usize, T: MontConfig<N>>(
) -> Option<SqrtPrecomputation<Fp<MontBackend<T, N>, N>>> {
    match T::MODULUS.mod_4() {
        3 => match T::MODULUS_PLUS_ONE_DIV_FOUR.as_ref() {
            Some(BigInt(modulus_plus_one_div_four)) => Some(SqrtPrecomputation::Case3Mod4 {
                modulus_plus_one_div_four,
            }),
            None => None,
        },
        _ => Some(SqrtPrecomputation::TonelliShanks {
            two_adicity: <MontBackend<T, N>>::TWO_ADICITY,
            quadratic_nonresidue_to_trace: T::TWO_ADIC_ROOT_OF_UNITY,
            trace_of_modulus_minus_one_div_two:
                &<Fp<MontBackend<T, N>, N>>::TRACE_MINUS_ONE_DIV_TWO.0,
        }),
    }
}

/// Adapted the `bn256-table` feature from `halo2curves`:
/// https://github.com/privacy-scaling-explorations/halo2curves/blob/main/script/bn256.py
pub const fn small_element_montgomery_precomputation<const N: usize, T: MontConfig<N>>(
) -> [Fp<MontBackend<T, N>, N>; PRECOMP_TABLE_SIZE] {
    let mut lookup_table: [Fp<MontBackend<T, N>, N>; PRECOMP_TABLE_SIZE] =
        [Fp::new_unchecked(BigInt::zero()); PRECOMP_TABLE_SIZE];

    let mut i: usize = 1;
    while i < PRECOMP_TABLE_SIZE {
        let mut limbs = [0u64; N];
        limbs[0] = i as u64;
        lookup_table[i] = <Fp<MontBackend<T, N>, N>>::new(BigInt::new(limbs));
        i += 1;
    }
    lookup_table
}

/// Construct a [`Fp<MontBackend<T, N>, N>`] element from a literal string. This
/// should be used primarily for constructing constant field elements; in a
/// non-const context, [`Fp::from_str`](`ark_std::str::FromStr::from_str`) is
/// preferable.
///
/// # Panics
///
/// If the integer represented by the string cannot fit in the number
/// of limbs of the `Fp`, this macro results in a
/// * compile-time error if used in a const context
/// * run-time error otherwise.
///
/// # Usage
///
/// ```rust
/// # use ark_test_curves::MontFp;
/// # use ark_test_curves::bls12_381 as ark_bls12_381;
/// # use ark_std::{One, str::FromStr};
/// use ark_bls12_381::Fq;
/// const ONE: Fq = MontFp!("1");
/// const NEG_ONE: Fq = MontFp!("-1");
///
/// fn check_correctness() {
///     assert_eq!(ONE, Fq::one());
///     assert_eq!(Fq::from_str("1").unwrap(), ONE);
///     assert_eq!(NEG_ONE, -Fq::one());
/// }
/// ```
#[macro_export]
macro_rules! MontFp {
    ($c0:expr) => {{
        let (is_positive, limbs) = $crate::ark_ff_macros::to_sign_and_limbs!($c0);
        $crate::Fp::from_sign_and_limbs(is_positive, &limbs)
    }};
}

pub use ark_ff_macros::MontConfig;

pub use MontFp;

pub struct MontBackend<T: MontConfig<N>, const N: usize>(PhantomData<T>);

impl<T: MontConfig<N>, const N: usize> FpConfig<N> for MontBackend<T, N> {
    /// The modulus of the field.
    const MODULUS: crate::BigInt<N> = T::MODULUS;

    /// A multiplicative generator of the field.
    /// `Self::GENERATOR` is an element having multiplicative order
    /// `Self::MODULUS - 1`.
    const GENERATOR: Fp<Self, N> = T::GENERATOR;

    /// Additive identity of the field, i.e. the element `e`
    /// such that, for all elements `f` of the field, `e + f = f`.
    const ZERO: Fp<Self, N> = Fp::new_unchecked(BigInt([0u64; N]));

    /// Multiplicative identity of the field, i.e. the element `e`
    /// such that, for all elements `f` of the field, `e * f = f`.
    const ONE: Fp<Self, N> = Fp::new_unchecked(T::R);

    const TWO_ADICITY: u32 = Self::MODULUS.two_adic_valuation();
    const TWO_ADIC_ROOT_OF_UNITY: Fp<Self, N> = T::TWO_ADIC_ROOT_OF_UNITY;
    const SMALL_SUBGROUP_BASE: Option<u32> = T::SMALL_SUBGROUP_BASE;
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = T::SMALL_SUBGROUP_BASE_ADICITY;
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<Fp<Self, N>> = T::LARGE_SUBGROUP_ROOT_OF_UNITY;
    const SQRT_PRECOMP: Option<crate::SqrtPrecomputation<Fp<Self, N>>> = T::SQRT_PRECOMP;
    const SMALL_ELEMENT_MONTGOMERY_PRECOMP: [Fp<Self, N>; PRECOMP_TABLE_SIZE] =
        T::SMALL_ELEMENT_MONTGOMERY_PRECOMP;

    fn add_assign(a: &mut Fp<Self, N>, b: &Fp<Self, N>) {
        T::add_assign(a, b)
    }

    fn sub_assign(a: &mut Fp<Self, N>, b: &Fp<Self, N>) {
        T::sub_assign(a, b)
    }

    fn double_in_place(a: &mut Fp<Self, N>) {
        T::double_in_place(a)
    }

    fn neg_in_place(a: &mut Fp<Self, N>) {
        T::neg_in_place(a)
    }

    /// This modular multiplication algorithm uses Montgomery
    /// reduction for efficient implementation. It also additionally
    /// uses the "no-carry optimization" outlined
    /// [here](https://hackmd.io/@zkteam/modular_multiplication) if
    /// `P::MODULUS` has (a) a non-zero MSB, and (b) at least one
    /// zero bit in the rest of the modulus.
    #[inline]
    fn mul_assign(a: &mut Fp<Self, N>, b: &Fp<Self, N>) {
        T::mul_assign(a, b)
    }

    fn sum_of_products<const M: usize>(a: &[Fp<Self, N>; M], b: &[Fp<Self, N>; M]) -> Fp<Self, N> {
        T::sum_of_products(a, b)
    }

    #[inline]
    #[allow(unused_braces, clippy::absurd_extreme_comparisons)]
    fn square_in_place(a: &mut Fp<Self, N>) {
        T::square_in_place(a)
    }

    fn inverse(a: &Fp<Self, N>) -> Option<Fp<Self, N>> {
        T::inverse(a)
    }

    fn from_bigint(r: BigInt<N>) -> Option<Fp<Self, N>> {
        T::from_bigint(r)
    }

    #[inline]
    #[allow(clippy::modulo_one)]
    fn into_bigint(a: Fp<Self, N>) -> BigInt<N> {
        T::into_bigint(a)
    }

    fn from_u64(r: u64) -> Option<Fp<Self, N>> {
        T::from_u64(r)
    }
}

impl<T: MontConfig<N>, const N: usize> Fp<MontBackend<T, N>, N> {
    #[doc(hidden)]
    pub const R: BigInt<N> = T::R;
    #[doc(hidden)]
    pub const R2: BigInt<N> = T::R2;
    #[doc(hidden)]
    pub const INV: u64 = T::INV;

    /// Construct a new field element from its underlying
    /// [`struct@BigInt`] data type.
    #[inline]
    pub const fn new(element: BigInt<N>) -> Self {
        let mut r = Self(element, PhantomData);
        if r.const_is_zero() {
            r
        } else {
            r = r.mul(&Fp(T::R2, PhantomData));
            r
        }
    }

    /// Construct a new field element from its underlying
    /// [`struct@BigInt`] data type.
    ///
    /// Unlike [`Self::new`], this method does not perform Montgomery reduction.
    /// Thus, this method should be used only when constructing
    /// an element from an integer that has already been put in
    /// Montgomery form.
    #[inline]
    pub const fn new_unchecked(element: BigInt<N>) -> Self {
        Self(element, PhantomData)
    }

    const fn const_is_zero(&self) -> bool {
        self.0.const_is_zero()
    }

    #[doc(hidden)]
    const fn const_neg(self) -> Self {
        if !self.const_is_zero() {
            Self::new_unchecked(Self::sub_with_borrow(&T::MODULUS, &self.0))
        } else {
            self
        }
    }

    /// Interpret a set of limbs (along with a sign) as a field element.
    /// For *internal* use only; please use the `ark_ff::MontFp` macro instead
    /// of this method
    #[doc(hidden)]
    pub const fn from_sign_and_limbs(is_positive: bool, limbs: &[u64]) -> Self {
        let mut repr = BigInt::<N>([0; N]);
        assert!(limbs.len() <= N);
        crate::const_for!((i in 0..(limbs.len())) {
            repr.0[i] = limbs[i];
        });
        let res = Self::new(repr);
        if is_positive {
            res
        } else {
            res.const_neg()
        }
    }

    const fn mul_without_cond_subtract(mut self, other: &Self) -> (bool, Self) {
        let (mut lo, mut hi) = ([0u64; N], [0u64; N]);
        crate::const_for!((i in 0..N) {
            let mut carry = 0;
            crate::const_for!((j in 0..N) {
                let k = i + j;
                if k >= N {
                    hi[k - N] = mac_with_carry!(hi[k - N], (self.0).0[i], (other.0).0[j], &mut carry);
                } else {
                    lo[k] = mac_with_carry!(lo[k], (self.0).0[i], (other.0).0[j], &mut carry);
                }
            });
            hi[i] = carry;
        });
        // Montgomery reduction
        let mut carry2 = 0;
        crate::const_for!((i in 0..N) {
            let tmp = lo[i].wrapping_mul(T::INV);
            let mut carry;
            mac!(lo[i], tmp, T::MODULUS.0[0], &mut carry);
            crate::const_for!((j in 1..N) {
                let k = i + j;
                if k >= N {
                    hi[k - N] = mac_with_carry!(hi[k - N], tmp, T::MODULUS.0[j], &mut carry);
                }  else {
                    lo[k] = mac_with_carry!(lo[k], tmp, T::MODULUS.0[j], &mut carry);
                }
            });
            hi[i] = adc!(hi[i], carry, &mut carry2);
        });

        crate::const_for!((i in 0..N) {
            (self.0).0[i] = hi[i];
        });
        (carry2 != 0, self)
    }

    const fn mul(self, other: &Self) -> Self {
        let (carry, res) = self.mul_without_cond_subtract(other);
        if T::MODULUS_HAS_SPARE_BIT {
            res.const_subtract_modulus()
        } else {
            res.const_subtract_modulus_with_carry(carry)
        }
    }

    #[inline(always)]
    pub fn mul_u64(self, other: u64) -> Self {
        // Stage 1: Bignum Multiplication
        // Compute c = self.0 * other. Result c has N+1 limbs.
        let c = bigint_mul_by_u64(&self.0 .0, other);

        // Stage 2: Barrett Reduction
        let r = barrett_reduce_nplus1_to_n::<T, N>(c);

        // Use the final r_n_limbs which holds the correct N-limb result
        Self::new_unchecked(BigInt::<N>(r))
    }

    /// Multiply by an i64. Invokes `mul_u64` if the input is positive,
    /// otherwise negates the result of `mul_u64` of the absolute value.
    #[inline(always)]
    pub fn mul_i64(self, other: i64) -> Self {
        if other >= 0 {
            // Multiply by the positive value directly
            self.mul_u64(other as u64)
        } else {
            // Multiply by the absolute value and then negate the result
            // (-other) cannot overflow since other is not i64::MIN
            -(self.mul_u64((-other) as u64))
        }
    }

    /// Multiply by an i128.
    /// Uses optimized mul_u64 if the absolute value of the input fits within u64,
    /// otherwise falls back to the two-step Barrett reduction (`mul_u128_aux`).
    #[inline(always)]
    pub fn mul_i128(self, other: i128) -> Self {
        if other >= 0 {
            let other_u128 = other as u128;
            if other_u128 <= u64::MAX as u128 {
                // Positive value fits in u64
                self.mul_u64(other_u128 as u64)
            } else {
                // Positive value requires u128 path
                self.mul_u128_aux(other_u128)
            }
        } else {
            // Negative value, compute absolute value as u128
            // (-other) will not overflow since other != i128::MIN (checked by fit condition)
            let abs_other = (-other) as u128;
            if abs_other <= u64::MAX as u128 {
                // Absolute value fits in u64
                -(self.mul_u64(abs_other as u64))
            } else {
                // Absolute value requires u128 path
                -(self.mul_u128_aux(abs_other))
            }
        }
    }

    /// Multiply by a u128.
    /// Uses optimized mul_u64 if the input fits within u64,
    /// otherwise falls back to standard multiplication.
    #[inline(always)]
    pub fn mul_u128(self, other: u128) -> Self {
        if other >> 64 == 0 {
            self.mul_u64(other as u64)
        } else {
            self.mul_u128_aux(other)
        }
    }

    /// Fallback option for mul_u128: if the input does not fit within u64,
    /// we perform a more expensive procedure with 2 rounds of Barrett reduction.
    #[inline(always)]
    pub fn mul_u128_aux(self, other: u128) -> Self {
        // Stage 1: Bignum Multiplication
        // Compute c = self.0 * other. Result c has N+2 limbs: (c_lo: [u64; 2], c_hi: [u64; N])
        let (c_lo, c_hi) = bigint_mul_by_u128(&self.0, other);

        // Stage 2: Two rounds of Barrett Reduction using the modular subroutine

        // Round 1: Reduce the top N+1 limbs c1 = floor(c / r).
        // c1 has low limb c_lo[1] and high N limbs c_hi.
        // Input to barrett_reduce is (u64, [u64; N]).
        let c1 = (c_lo[1], c_hi);
        let r1 = barrett_reduce_nplus1_to_n::<T, N>(c1); // r1 = c1 mod p ([u64; N])

        // Round 2: Reduce c2 = c_lo[0] + r1 * r.
        // c2 has low limb c_lo[0] and high N limbs r1.0.
        // Input to barrett_reduce is (u64, [u64; N]).
        let c2 = (c_lo[0], r1); // Pass r1 directly as the high N limbs array
        let r2 = barrett_reduce_nplus1_to_n::<T, N>(c2); // r2 = c2 mod p = c mod p ([u64; N])

        Self::new_unchecked(BigInt::<N>(r2))
    }

    const fn const_is_valid(&self) -> bool {
        crate::const_for!((i in 0..N) {
            if (self.0).0[N - i - 1] < T::MODULUS.0[N - i - 1] {
                return true
            } else if (self.0).0[N - i - 1] > T::MODULUS.0[N - i - 1] {
                return false
            }
        });
        false
    }

    #[inline]
    const fn const_subtract_modulus(mut self) -> Self {
        if !self.const_is_valid() {
            self.0 = Self::sub_with_borrow(&self.0, &T::MODULUS);
        }
        self
    }

    #[inline]
    const fn const_subtract_modulus_with_carry(mut self, carry: bool) -> Self {
        if carry || !self.const_is_valid() {
            self.0 = Self::sub_with_borrow(&self.0, &T::MODULUS);
        }
        self
    }

    const fn sub_with_borrow(a: &BigInt<N>, b: &BigInt<N>) -> BigInt<N> {
        a.const_sub_with_borrow(b).0
    }
}

/// Multiply a N-limb big integer with a u64, producing a N+1 limb result,
/// represented as a tuple of a u64 low limb and an array of N high limbs.
#[unroll_for_loops(8)]
#[inline(always)]
fn bigint_mul_by_u64<const N: usize>(val: &[u64; N], other: u64) -> (u64, [u64; N]) {
    let mut result_hi = [0u64; N];
    let mut carry: u64; // Start with carry = 0

    // Calculate the full 128-bit product of the lowest limb
    let prod_lo: u128 = (val[0] as u128) * (other as u128);
    let result_lo = prod_lo as u64; // Lowest limb of the result
    carry = (prod_lo >> 64) as u64; // Carry into the high part

    // Iterate through the remaining limbs of the input BigInt
    for i in 1..N {
        // Calculate the full 128-bit product of the current limb and the u64 multiplier
        let prod_hi: u128 = (val[i] as u128) * (other as u128) + (carry as u128);
        result_hi[i - 1] = prod_hi as u64; // Store in result_hi[0] to result_hi[N-2]
        carry = (prod_hi >> 64) as u64;
    }

    // After the loop, the final carry is the highest limb (N-th limb of the high part)
    result_hi[N - 1] = carry;

    (result_lo, result_hi)
}

/// Multiply a N+1 limb big integer (represented as low N limbs and high u64) with a u64,
/// producing a N+1 limb result in the same format.
/// Also returns a boolean indicating if there was a carry out (overflow).
#[unroll_for_loops(8)]
#[inline(always)]
fn bigint_plus_one_mul_by_u64<const N: usize>(
    val: ([u64; N], u64),
    other: u64,
) -> (([u64; N], u64), bool) {
    let (val_lo_n, val_hi) = val;
    let mut result_lo_n = [0u64; N];
    let mut carry: u128 = 0; // Use u128 for intermediate carry

    // Stage 1: Multiply the low N limbs
    for i in 0..N {
        let prod: u128 = (val_lo_n[i] as u128) * (other as u128) + carry;
        result_lo_n[i] = prod as u64;
        carry = prod >> 64;
    }

    // Stage 2: Multiply the high limb
    let prod_hi: u128 = (val_hi as u128) * (other as u128) + carry;
    let result_hi = prod_hi as u64;
    let final_carry = prod_hi >> 64;

    // Final carry indicates overflow
    let overflow = final_carry != 0;
    ((result_lo_n, result_hi), overflow)
}

/// Subtract two N+1 limb big integers (represented as low N limbs and high u64).
/// Returns the N+1 limb result and a boolean indicating if a borrow occurred.
#[unroll_for_loops(8)]
#[inline(always)]
fn sub_bigint_plus_one<const N: usize>(
    a: ([u64; N], u64),
    b: ([u64; N], u64),
) -> (([u64; N], u64), bool) {
    let (mut a_lo_n, mut a_hi) = a;
    let (b_lo_n, b_hi) = b;
    let mut borrow: u64 = 0; // sbb uses u64 for borrow

    // Subtract low N limbs
    for i in 0..N {
        // Updates a_lo_n[i] in place and returns the new borrow
        borrow = fa::sbb(&mut a_lo_n[i], b_lo_n[i], borrow);
    }

    // Subtract high u64 limb
    borrow = fa::sbb(&mut a_hi, b_hi, borrow);

    // Final borrow indicates if the result is negative (b > a)
    let final_borrow_occurred = borrow != 0;

    ((a_lo_n, a_hi), final_borrow_occurred)
}

/// Subtract two N+1 limb big integers where `a` is (u64, [u64; N]) and `b` is ([u64; N], u64).
/// Returns the N+1 limb result as ([u64; N], u64) and a boolean indicating if a borrow occurred.
#[unroll_for_loops(8)]
#[inline(always)]
fn sub_bigint_plus_one_prime<const N: usize>(
    a: (u64, [u64; N]), // Format: (low_limb, high_n_limbs)
    b: ([u64; N], u64), // Format: (low_n_limbs, high_limb)
) -> (([u64; N], u64), bool) {
    let (a_lo, a_hi_n) = a;
    let (b_lo_n, b_hi) = b;
    let mut result_lo_n = [0u64; N];
    let mut borrow: u64 = 0;

    // Subtract low limb: result_lo_n[0] = a_lo - b_lo_n[0] - borrow (initial borrow = 0)
    result_lo_n[0] = a_lo; // Initialize result limb with a_lo
    borrow = fa::sbb(&mut result_lo_n[0], b_lo_n[0], borrow); // result_lo_n[0] -= b_lo_n[0] + borrow

    // Subtract middle limbs (if N > 1): result_lo_n[i] = a_hi_n[i-1] - b_lo_n[i] - borrow
    // This loop covers indices i = 1 to N-1.
    // It uses a_hi_n limbs from index 0 to N-2.
    for i in 1..N {
        result_lo_n[i] = a_hi_n[i - 1]; // Initialize result limb with corresponding a limb
        borrow = fa::sbb(&mut result_lo_n[i], b_lo_n[i], borrow); // result_lo_n[i] -= b_lo_n[i] + borrow
    }

    // Subtract high limb: result_hi = a_hi_n[N-1] - b_hi - borrow
    let mut result_hi = a_hi_n[N - 1]; // Initialize result limb with last a limb
    borrow = fa::sbb(&mut result_hi, b_hi, borrow); // result_hi -= b_hi + borrow

    let final_borrow_occurred = borrow != 0;

    ((result_lo_n, result_hi), final_borrow_occurred)
}

/// Compare two N+1 limb big integers (represented as low N limbs and high u64).
#[unroll_for_loops(8)]
#[inline(always)]
fn compare_bigint_plus_one<const N: usize>(
    a: ([u64; N], u64),
    b: ([u64; N], u64),
) -> core::cmp::Ordering {
    // Compare high u64 limb first
    if a.1 > b.1 {
        return core::cmp::Ordering::Greater;
    } else if a.1 < b.1 {
        return core::cmp::Ordering::Less;
    }
    // High limbs are equal, compare the low N limbs from most significant (N-1) down to 0
    for i in (0..N).rev() {
        if a.0[i] > b.0[i] {
            return core::cmp::Ordering::Greater;
        } else if a.0[i] < b.0[i] {
            return core::cmp::Ordering::Less;
        }
    }
    // All limbs are equal
    return core::cmp::Ordering::Equal;
}

/// Multiply a N-limb big integer with a u128, producing a N+2 limb result,
/// represented as a tuple of an array of 2 low limbs and an array of N high limbs.
#[unroll_for_loops(8)]
#[inline(always)]
fn bigint_mul_by_u128<const N: usize>(val: &BigInt<N>, other: u128) -> ([u64; 2], [u64; N]) {
    let other_lo = other as u64;
    let other_hi = (other >> 64) as u64;

    // Compute partial products
    // p1 = val * other_lo -> (N+1) limbs: (p1_lo: u64, p1_hi: [u64; N])
    let (p1_lo, p1_hi) = bigint_mul_by_u64(&val.0, other_lo);
    // p2 = val * other_hi -> (N+1) limbs: (p2_lo: u64, p2_hi: [u64; N])
    let (p2_lo, p2_hi) = bigint_mul_by_u64(&val.0, other_hi);

    // Calculate the final result r = p1 + (p2 << 64) limb by limb.
    // p1       : [p1_lo, p1_hi[0], ..., p1_hi[N-1]]
    // p2 << 64 : [0, p2_lo, p2_hi[0], ..., p2_hi[N-1]]
    // Sum (r)  : [r_lo[0], r_lo[1], r_hi[0], ..., r_hi[N-1]] (N+2 limbs)

    let mut r_lo = [0u64; 2];
    let mut r_hi = [0u64; N];
    let mut carry: u64 = 0;

    // r_lo[0] = p1_lo + 0 + carry (carry is initially 0)
    r_lo[0] = p1_lo;
    // carry = 0; // Initial carry is 0

    // Calculate r_lo[1] = p1_hi[0] + p2_lo + carry (limb 1)
    r_lo[1] = p1_hi[0]; // Initialize with p1 limb
    carry = fa::adc(&mut r_lo[1], p2_lo, carry); // Add p2 limb and carry

    // Calculate r_hi[0] to r_hi[N-1] (limbs 2 to N+1)
    for i in 0..N {
        let p1_limb = if i + 1 < N { p1_hi[i + 1] } else { 0 }; // Limb p1[i+2]
        let p2_limb = p2_hi[i]; // Limb p2[i+1]

        // r_hi[i] = p1_limb + p2_limb + carry
        r_hi[i] = p1_limb; // Initialize with p1 limb
        carry = fa::adc(&mut r_hi[i], p2_limb, carry); // Add p2 limb and carry
    }

    // The final carry MUST be zero for the result to fit in N+2 limbs.
    debug_assert!(carry == 0, "Overflow in bigint_mul_by_u128");

    (r_lo, r_hi)
}

/// Old conditional subtraction logic for Barrett reduction
/// Takes an N+1 limb intermediate result `r_tmp` and returns the N-limb final result.
#[inline(always)]
fn _barrett_cond_subtract_old<T: MontConfig<N>, const N: usize>(
    r_tmp: ([u64; N], u64),
) -> [u64; N] {
    let mut current_r = r_tmp; // Working variable in ([u64; N], u64) format

    if T::MODULUS_NUM_SPARE_BITS >= 1 {
        // Case S >= 1
        if T::MODULUS_NUM_SPARE_BITS >= 2 {
            // Optimization for S >= 2: r_tmp = c - m*2p < 4p
            // High limb of current_r should initially be 0
            debug_assert!(
                current_r.1 == 0,
                "High limb of r_tmp should be zero when S >= 2"
            );

            // Conditional subtraction 1 (if r >= 2P) using N+1 compare/sub
            if compare_bigint_plus_one(current_r, T::MODULUS_TIMES_2_NPLUS1)
                != core::cmp::Ordering::Less
            {
                let (sub_res, sub_borrow) =
                    sub_bigint_plus_one(current_r, T::MODULUS_TIMES_2_NPLUS1);
                debug_assert!(!sub_borrow, "Borrow should not occur subtracting 2P (S>=2)");
                current_r = sub_res;
            }
            // Conditional subtraction 2 (if r >= P) using N+1 compare/sub
            if compare_bigint_plus_one(current_r, T::MODULUS_NPLUS1) != core::cmp::Ordering::Less {
                let (sub_res, sub_borrow) = sub_bigint_plus_one(current_r, T::MODULUS_NPLUS1);
                debug_assert!(!sub_borrow, "Borrow should not occur subtracting P (S>=2)");
                current_r = sub_res;
            }
            // Result must fit in N limbs now
            debug_assert!(
                current_r.1 == 0,
                "High limb must be zero after final subtraction (S>=2)"
            );
            current_r.0 // Return N low limbs
        } else {
            // Case S == 1: r_tmp = c - m*2p might temporarily exceed N limbs initially

            // Conditional subtraction 1: if r >= 2p
            if compare_bigint_plus_one(current_r, T::MODULUS_TIMES_2_NPLUS1)
                != core::cmp::Ordering::Less
            {
                // Subtract 2P using N+1 limb subtraction
                let (sub_res, sub_borrow) =
                    sub_bigint_plus_one(current_r, T::MODULUS_TIMES_2_NPLUS1);
                // After subtracting 2P, the result MUST fit in N limbs
                debug_assert!(
                    sub_res.1 == 0,
                    "High limb must be 0 after 2P subtraction when S=1"
                );
                debug_assert!(
                    !sub_borrow,
                    "Borrow should not occur when subtracting 2P for S=1"
                );
                current_r = sub_res; // Update current_r (now guaranteed to have high limb 0)
            }
            // At this point, current_r < 2P and its high limb is 0.
            debug_assert!(
                current_r.1 == 0,
                "High limb should be 0 before N-limb subtraction (S=1)"
            );
            let mut r_n_limbs = BigInt::<N>(current_r.0); // Extract N low limbs

            // Conditional subtraction 2 (if r >= P) using N limbs directly
            if r_n_limbs >= T::MODULUS {
                // Compare N limbs
                r_n_limbs.sub_with_borrow(&T::MODULUS); // Subtract N limbs. Ignore borrow.
            }
            r_n_limbs.0 // Return N low limbs
        }
    } else {
        // Case S == 0: Use (N+1)-limb helpers throughout

        // Conditional subtraction 1: if r >= 2p
        if compare_bigint_plus_one(current_r, T::MODULUS_TIMES_2_NPLUS1)
            != core::cmp::Ordering::Less
        {
            let (sub_res, sub_borrow) = sub_bigint_plus_one(current_r, T::MODULUS_TIMES_2_NPLUS1);
            debug_assert!(!sub_borrow, "Borrow should not occur subtracting 2P (S=0)");
            current_r = sub_res;
        }
        // Now current_r = c mod 2p, represented as ([u64; N], u64)

        // Conditional subtraction 2: if r >= p
        if compare_bigint_plus_one(current_r, T::MODULUS_NPLUS1) != core::cmp::Ordering::Less {
            // if r >= p
            let (sub_res, sub_borrow) = sub_bigint_plus_one(current_r, T::MODULUS_NPLUS1);
            // Result MUST fit in N limbs now
            debug_assert!(
                sub_res.1 == 0,
                "High limb must be zero after subtracting P (S=0)"
            );
            debug_assert!(
                !sub_borrow,
                "Borrow should not occur when subtracting P for S=0"
            );
            current_r = sub_res;
        }
        // At this point, current_r < P and its high limb must be 0.
        debug_assert!(
            current_r.1 == 0,
            "High limb must be zero after final subtraction (S=0)"
        );
        current_r.0 // Return N low limbs
    }
}

/// Conditional subtraction logic for Barrett reduction, trading an extra comparison for a conditional subtraction.
/// Includes optimizations based on MODULUS_NUM_SPARE_BITS.
/// Takes an N+1 limb intermediate result `r_tmp` (in `([u64; N], u64)` format) and returns the N-limb final result.
#[unroll_for_loops(4)]
#[inline(always)]
fn barrett_cond_subtract<T: MontConfig<N>, const N: usize>(r_tmp: ([u64; N], u64)) -> [u64; N] {
    let final_limbs: [u64; N];
    let r_n = BigInt::<N>(r_tmp.0); // N low limbs as BigInt
    let r_hi = r_tmp.1; // High limb

    // Compare with 2p
    let compare_2p = if T::MODULUS_NUM_SPARE_BITS == 0 {
        // S = 0: Must use N+1 compare
        compare_bigint_plus_one(r_tmp, T::MODULUS_TIMES_2_NPLUS1)
    } else {
        // S >= 1: 2p fits N limbs (mostly). Compare N limbs.
        // We assume r_tmp's high limb is 0 here if S >= 1.
        debug_assert!(
            r_hi == 0,
            "High limb expected to be 0 if S >= 1 before 2p comparison"
        );
        let p2_n = BigInt::<N>(T::MODULUS_TIMES_2_NPLUS1.0);
        r_n.cmp(&p2_n)
    };

    if compare_2p != core::cmp::Ordering::Less {
        // r_tmp >= 2p
        // Compare with 3p
        let compare_3p = if T::MODULUS_NUM_SPARE_BITS < 2 {
            // S < 2 (S=0 or S=1): Need N+1 compare
            compare_bigint_plus_one(r_tmp, T::MODULUS_TIMES_3_NPLUS1)
        } else {
            // S >= 2: 3p fits N limbs. Compare N limbs.
            debug_assert!(
                r_hi == 0,
                "High limb expected to be 0 if S >= 2 before 3p comparison"
            );
            let p3_n = BigInt::<N>(T::MODULUS_TIMES_3_NPLUS1.0);
            r_n.cmp(&p3_n)
        };

        if compare_3p != core::cmp::Ordering::Less {
            // r_tmp >= 3p
            // Subtract 3p
            if T::MODULUS_NUM_SPARE_BITS >= 2 {
                // S >= 2: 3p fits N limbs. Use N-limb sub.
                debug_assert!(
                    r_hi == 0,
                    "High limb expected to be 0 if S >= 2 for 3p subtraction"
                );
                let p3_n = BigInt::<N>(T::MODULUS_TIMES_3_NPLUS1.0);
                let (res_n, borrow_n) = r_n.const_sub_with_borrow(&p3_n);
                debug_assert!(!borrow_n, "Borrow should not occur subtracting 3p (S>=2)");
                final_limbs = res_n.0;
            } else {
                // S < 2: Use N+1 limb sub.
                let ((res_n_limbs, res_hi_limb), borrow_n1) =
                    sub_bigint_plus_one(r_tmp, T::MODULUS_TIMES_3_NPLUS1);
                debug_assert!(!borrow_n1, "Borrow should not occur subtracting 3p (S<2)");
                debug_assert!(
                    res_hi_limb == 0,
                    "High limb must be zero after subtracting 3p"
                );
                final_limbs = res_n_limbs;
            }
        } else {
            // 2p <= r_tmp < 3p
            // Subtract 2p
            if T::MODULUS_NUM_SPARE_BITS >= 1 {
                // S >= 1: 2p fits N limbs (mostly). Use N-limb sub.
                debug_assert!(
                    r_hi == 0,
                    "High limb expected to be 0 if S >= 1 for 2p subtraction"
                );
                let p2_n = BigInt::<N>(T::MODULUS_TIMES_2_NPLUS1.0);
                let (res_n, borrow_n) = r_n.const_sub_with_borrow(&p2_n);
                debug_assert!(!borrow_n, "Borrow should not occur subtracting 2p (S>=1)");
                final_limbs = res_n.0;
            } else {
                // S == 0: Use N+1 limb sub.
                let ((res_n_limbs, res_hi_limb), borrow_n1) =
                    sub_bigint_plus_one(r_tmp, T::MODULUS_TIMES_2_NPLUS1);
                debug_assert!(!borrow_n1, "Borrow should not occur subtracting 2p (S=0)");
                debug_assert!(
                    res_hi_limb == 0,
                    "High limb must be zero after subtracting 2p"
                );
                final_limbs = res_n_limbs;
            }
        }
    } else {
        // r_tmp < 2p
        // Compare with p
        let compare_p = if T::MODULUS_NUM_SPARE_BITS >= 1 {
            // S >= 1: Use N-limb compare.
            // Assume r_tmp high limb is 0 because r_tmp < 2p and 2p fits N limbs (mostly) if S >= 1
            debug_assert!(r_hi == 0, "High limb expected to be 0 if S >= 1 and r < 2p");
            r_n.cmp(&T::MODULUS) // Compare N limbs
        } else {
            // S == 0: Use N+1 limb compare.
            compare_bigint_plus_one(r_tmp, T::MODULUS_NPLUS1)
        };

        if compare_p != core::cmp::Ordering::Less {
            // p <= r_tmp < 2p
            // Subtract p
            if T::MODULUS_NUM_SPARE_BITS >= 1 {
                // S >= 1: Use N-limb sub.
                debug_assert!(
                    r_hi == 0,
                    "High limb expected to be 0 if S >= 1 for p subtraction"
                );
                let (res_n, borrow_n) = r_n.const_sub_with_borrow(&T::MODULUS);
                debug_assert!(!borrow_n, "Borrow should not occur subtracting p (S>=1)");
                final_limbs = res_n.0;
            } else {
                // S == 0: Use N+1 limb sub.
                let ((res_n_limbs, res_hi_limb), borrow_n1) =
                    sub_bigint_plus_one(r_tmp, T::MODULUS_NPLUS1);
                debug_assert!(!borrow_n1, "Borrow should not occur subtracting p (S=0)");
                debug_assert!(
                    res_hi_limb == 0,
                    "High limb must be zero after subtracting p"
                );
                final_limbs = res_n_limbs;
            }
        } else {
            // r_tmp < p
            // Subtract 0 (No-op)
            // Result must already fit in N limbs. Assert high limb is 0.
            debug_assert!(r_hi == 0, "High limb must be zero when r_tmp < p");
            final_limbs = r_n.0; // Use the low N limbs directly
        }
    }
    final_limbs
}

/// Helper function to perform Barrett reduction from N+1 limbs to N limbs.
/// Input `c` is represented as `(u64, [u64; N])` (to be compatible with outside invocations).
/// Internally, it converts to `([u64; N], u64)` and operates in that format.
/// Output is the N-limb result `[u64; N]`.
#[unroll_for_loops(4)]
#[inline(always)]
fn barrett_reduce_nplus1_to_n<T: MontConfig<N>, const N: usize>(c: (u64, [u64; N])) -> [u64; N] {
    let (c_lo, c_hi) = c; // c_lo is the lowest limb, c_hi holds the top N limbs

    // Compute tilde_c = floor(c / R') = floor(c / 2^MODULUS_BITS)
    // This involves the top two limbs of the N+1 limb number `c`.
    // The highest limb is c_hi[N-1]. The second highest is c_hi[N-2].
    // Assume that `N >= 1`
    let tilde_c: u64 = if T::MODULUS_HAS_SPARE_BIT {
        let high_limb = c_hi[N - 1];
        let second_high_limb = if N > 1 { c_hi[N - 2] } else { c_lo }; // Use c_lo if N=1
        (high_limb << T::MODULUS_NUM_SPARE_BITS)
            + (second_high_limb >> (64 - T::MODULUS_NUM_SPARE_BITS))
    } else {
        c_hi[N - 1] // If no spare bits, tilde_c is just the highest limb
    };

    // Estimate m = floor( (tilde_c * BARRETT_MU) / r )
    // where r = 2^64
    let m: u64 = ((tilde_c as u128 * T::BARRETT_MU as u128) >> 64) as u64;

    // Compute m * 2p (N+1 limbs)
    let (m_times_2p, m2p_overflow) = bigint_plus_one_mul_by_u64::<N>(T::MODULUS_TIMES_2_NPLUS1, m);
    // If m * 2p overflows N+1 limbs, the logic might be flawed or input c was too large.
    debug_assert!(!m2p_overflow, "Overflow calculating m * 2p");

    // Compute r_tmp = c - m * 2p (result is ([u64; N], u64))
    let (r_tmp, r_tmp_borrow) = sub_bigint_plus_one_prime(c, m_times_2p);
    // A borrow here implies c was smaller than m*2p, which shouldn't happen with correct m.
    debug_assert!(!r_tmp_borrow, "Borrow occurred calculating c - m*2p");

    // Use the optimized conditional subtraction which expects ([u64; N], u64)
    barrett_cond_subtract::<T, N>(r_tmp)
}

#[cfg(test)]
mod test {
    use ark_std::{str::FromStr, test_rng, vec::*, UniformRand};
    use ark_test_curves::ark_ff::BigInt as ArkBigInt;
    use ark_test_curves::bn254::Fr;
    use num_bigint::{BigInt, BigUint, Sign};
    use rand::Rng;

    #[test]
    fn test_mul_u64_random() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let a = Fr::rand(&mut rng);
            let b_val: u64 = rng.gen();

            // Expected result using standard field multiplication
            let b_bigint = ArkBigInt::from(b_val);
            let expected_c = a * Fr::from(b_bigint);

            // Actual result using the function under test
            let result_c = a.mul_u64(b_val);

            assert_eq!(
                result_c,
                expected_c,
                "mul_u64 failed: a = {}, b = {}\\nGot:      {}\\nExpected: {}\\nGot (Debug):      {:?}\\nExpected (Debug): {:?}", // Added Debug formatting
                a, b_val, result_c, expected_c, result_c, expected_c // Added vars for Debug
            );
        }
    }

    #[test]
    fn test_mul_i64_random() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let a = Fr::rand(&mut rng);
            let b_val: i64 = rng.gen();

            // Expected result using standard field multiplication
            let expected_c = if b_val >= 0 {
                let b_bigint = ArkBigInt::from(b_val as u64);
                a * Fr::from(b_bigint)
            } else {
                let b_bigint = ArkBigInt::from(-b_val as u64);
                -(a * Fr::from(b_bigint))
            };

            // Actual result using the function under test
            let result_c = a.mul_i64(b_val);

            assert_eq!(
                result_c, expected_c,
                "mul_i64 failed: a = {}, b = {}\nGot:      {}\nExpected: {}",
                a, b_val, result_c, expected_c
            );
        }
    }

    #[test]
    fn test_mul_u128_random() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let a = Fr::rand(&mut rng);
            let b_val: u128 = rng.gen();

            // Expected result using standard field multiplication
            let b_bigint = ArkBigInt::from(b_val); // Convert u128 -> BigInt
            let expected_c = a * Fr::from(b_bigint);

            // Actual result using the function under test
            let result_c = a.mul_u128(b_val);

            assert_eq!(
                result_c, expected_c,
                "mul_u128 failed: a = {}, b = {}\nGot:      {}\nExpected: {}",
                a, b_val, result_c, expected_c
            );
        }
    }

    #[test]
    fn test_mul_i128_random() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let a = Fr::rand(&mut rng);
            // Avoid i128::MIN as negation overflows
            let b_val: i128 = rng.gen_range(i128::MIN + 1..=i128::MAX);

            // Expected result using standard field multiplication
            let expected_c = if b_val >= 0 {
                let b_bigint = ArkBigInt::from(b_val as u128);
                a * Fr::from(b_bigint)
            } else {
                let b_bigint = ArkBigInt::from((-b_val) as u128);
                -(a * Fr::from(b_bigint))
            };

            // Actual result using the function under test
            let result_c = a.mul_i128(b_val);

            assert_eq!(
                result_c, expected_c,
                "mul_i128 failed: a = {}, b = {}\nGot:      {}\nExpected: {}",
                a, b_val, result_c, expected_c
            );
        }
    }

    #[test]
    fn test_mont_macro_correctness() {
        // This test succeeds **only** on the secp256k1 curve.
        let (is_positive, limbs) = str_to_limbs_u64(
            "111192936301596926984056301862066282284536849596023571352007112326586892541694",
        );
        let t = Fr::from_sign_and_limbs(is_positive, &limbs);

        let result: BigUint = t.into();
        let expected = BigUint::from_str(
            "111192936301596926984056301862066282284536849596023571352007112326586892541694",
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    fn str_to_limbs_u64(num: &str) -> (bool, Vec<u64>) {
        let (sign, digits) = BigInt::from_str(num)
            .expect("could not parse to bigint")
            .to_radix_le(16);
        let limbs = digits
            .chunks(16)
            .map(|chunk| {
                let mut this = 0u64;
                for (i, hexit) in chunk.iter().enumerate() {
                    this += (*hexit as u64) << (4 * i);
                }
                this
            })
            .collect::<Vec<_>>();

        let sign_is_positive = sign != Sign::Minus;
        (sign_is_positive, limbs)
    }
}
