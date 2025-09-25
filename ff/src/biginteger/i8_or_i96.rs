use crate::biginteger::{S160, S224};
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use allocative::Allocative;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};

/// Compact signed integer optimized for the common `i8` case, widening to a 96-bit
/// split representation when needed (low 64 bits in `large_lo`, next 32 bits in `large_hi`).
///
/// ## Design goals:
/// - Set fields so that this fits in 16 bytes
/// - Encode the vast majority of values as `i8` for space/time locality.
/// - Keep all operations `const fn` so macros and static tables can fold at compile-time.
/// - After every operation, results are canonicalized to the smallest fitting form:
///   if a result fits in `i8`, it is stored in `small_i8`; otherwise it is stored
///   as a 96-bit split in `large_lo`/`large_hi`.
///
/// ## Layout and Semantics
///
/// The 96-bit value is stored in two's complement format, split across two fields:
/// - `large_hi: i32`: The upper 32 bits, which includes the sign bit of the 96-bit integer.
/// - `large_lo: u64`: The lower 64 bits, treated as an unsigned block of bits.
///
/// The full value can be reconstructed using the formula:
/// `value = (large_hi as i128) << 64 | (large_lo as i128)`
/// This is equivalent to sign-extending `large_hi` and zero-extending `large_lo`.
///
/// ## Notes:
/// - Arithmetic uses `i128` for intermediate computation but the representation is signed 96-bit.
///   Results are canonicalized to the smallest fitting form. If a result does not fit in 96 bits,
///   it is truncated to signed 96-bit two's complement (wrapping modulo 2^96).
/// - The `neg` implementation avoids `i8` overflow by widening `i8::MIN` to the wide form.
/// - Conversions are total: `to_i128()` always returns the exact value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Allocative)]
pub struct I8OrI96 {
    /// The lower 64 bits of the constant value.
    large_lo: u64,
    /// The upper 32 (signed) bits above `large_lo` (bits 64..95)
    large_hi: i32,
    /// Small constants that fit in i8 (-128 to 127)
    pub small_i8: i8,
    /// Whether the constant value is small (i8)
    pub is_small: bool,
}

impl I8OrI96 {
    /// Returns zero encoded as `I8(0)`.
    pub const fn zero() -> Self {
        I8OrI96 {
            large_lo: 0,
            large_hi: 0,
            is_small: true,
            small_i8: 0,
        }
    }

    /// Returns one encoded as `I8(1)`.
    pub const fn one() -> Self {
        I8OrI96 {
            large_lo: 0,
            large_hi: 0,
            is_small: true,
            small_i8: 1,
        }
    }

    /// Construct from `i8` without widening.
    pub const fn from_i8(value: i8) -> Self {
        I8OrI96 {
            large_lo: 0,
            large_hi: 0,
            is_small: true,
            small_i8: value,
        }
    }

    /// Construct from `i128`, canonicalizing to `I8` if it fits.
    /// Assumes the value fits in 96 bits (i64 + i32)
    pub const fn from_i128(value: i128) -> Self {
        if value >= i8::MIN as i128 && value <= i8::MAX as i128 {
            I8OrI96 {
                large_lo: 0,
                large_hi: 0,
                is_small: true,
                small_i8: value as i8,
            }
        } else {
            // Store as 96-bit signed split: low 64 bits and next 32 bits
            I8OrI96 {
                large_lo: value as u64,
                large_hi: (value >> 64) as i32,
                is_small: false,
                small_i8: 0,
            }
        }
    }

    /// Mutate in-place from `i8` without widening. Only updates `small_i8` and `is_small`.
    #[inline]
    pub const fn set_from_i8(&mut self, value: i8) {
        self.small_i8 = value;
        self.is_small = true;
    }

    /// Mutate in-place from `i128`, canonicalizing to `i8` when it fits.
    /// Assumes the value fits in 96 bits (i64 + i32). Minimizes writes.
    #[inline]
    pub const fn set_from_i128(&mut self, value: i128) {
        if value >= i8::MIN as i128 && value <= i8::MAX as i128 {
            self.small_i8 = value as i8;
            self.is_small = true;
        } else {
            self.large_lo = value as u64;
            self.large_hi = (value >> 64) as i32;
            self.is_small = false;
        }
    }

    /// Exact conversion to `i128`.
    #[inline]
    pub const fn to_i128(&self) -> i128 {
        if self.is_small {
            self.small_i8 as i128
        } else {
            // The `large_lo` (u64) is zero-extended to i128, and `large_hi` (i32) is sign-extended.
            // This correctly reconstructs the 96-bit signed value.
            (self.large_lo as i128) | ((self.large_hi as i128) << 64)
        }
    }

    /// Absolute value as unsigned magnitude.
    pub const fn unsigned_abs(&self) -> u128 {
        let v = self.to_i128();
        v.unsigned_abs()
    }

    /// Returns true if the value equals zero.
    #[inline]
    pub const fn is_zero(&self) -> bool {
        if self.is_small {
            self.small_i8 == 0
        } else {
            self.large_lo == 0 && self.large_hi == 0
        }
    }

    /// Returns true if the value is encoded as `I128`.
    #[inline]
    pub const fn is_large(&self) -> bool {
        !self.is_small
    }

    /// Add two constants, returning a canonicalized result.
    ///
    /// Fast-path: if both operands are `I8`, perform `i8` addition directly.
    /// If the `i8` addition overflows, it falls back to the `i128` slow path.
    #[inline]
    pub const fn add(self, other: I8OrI96) -> I8OrI96 {
        let mut out = self;
        out.add_assign(&other);
        out
    }

    /// In-place addition assignment: `self = self + other`.
    /// Preserves fast path and falls back to `i128` on `i8` overflow.
    #[inline]
    pub const fn add_assign(&mut self, other: &I8OrI96) {
        if self.is_small && other.is_small {
            let (sum, overflow) = self.small_i8.overflowing_add(other.small_i8);
            if !overflow {
                self.set_from_i8(sum);
                return;
            }
        }
        let sum = self.to_i128() + other.to_i128();
        self.set_from_i128(sum);
    }

    /// Multiply two constants, returning a canonicalized result.
    ///
    /// Fast-path: if both operands are `I8`, perform `i8` multiplication directly.
    /// If `i8` multiplication overflows, it falls back to the `i128` slow path.
    #[inline]
    pub const fn mul(self, other: I8OrI96) -> I8OrI96 {
        let mut out = self;
        out.mul_assign(&other);
        out
    }

    /// In-place multiplication assignment: `self = self * other`.
    /// Preserves fast path and falls back to `i128` on `i8` overflow.
    #[inline]
    pub const fn mul_assign(&mut self, other: &I8OrI96) {
        if self.is_small && other.is_small {
            let (prod, overflow) = self.small_i8.overflowing_mul(other.small_i8);
            if !overflow {
                self.set_from_i8(prod);
                return;
            }
        }
        let prod = self.to_i128() * other.to_i128();
        self.set_from_i128(prod);
    }

    /// Arithmetic negation with canonicalization.
    ///
    /// Special-cases `I8(i8::MIN)` to avoid overflow by widening to `I128`.
    /// In-place arithmetic negation. Preserves `i8::MIN` widening behavior.
    #[inline]
    pub const fn neg(&mut self) {
        if self.is_small {
            let v = self.small_i8;
            if v == i8::MIN {
                self.set_from_i128(-(v as i128));
            } else {
                self.set_from_i8(-v);
            }
        } else {
            self.set_from_i128(-self.to_i128());
        }
    }

    /// Subtraction returning a new value. Delegates to `sub_assign`.
    #[inline]
    pub const fn sub(self, other: I8OrI96) -> I8OrI96 {
        let mut out = self;
        out.sub_assign(&other);
        out
    }

    /// In-place subtraction assignment: `self = self - other`.
    /// Fast-path: if both operands are `I8`, perform `i8` subtraction directly.
    /// If `i8` subtraction overflows, it falls back to `i128` slow path.
    #[inline]
    pub const fn sub_assign(&mut self, other: &I8OrI96) {
        if self.is_small && other.is_small {
            let (diff, overflow) = self.small_i8.overflowing_sub(other.small_i8);
            if !overflow {
                self.set_from_i8(diff);
                return;
            }
        }
        let diff = self.to_i128() - other.to_i128();
        self.set_from_i128(diff);
    }
}

impl Add for I8OrI96 {
    type Output = I8OrI96;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        I8OrI96::add(self, rhs)
    }
}

impl AddAssign for I8OrI96 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs)
    }
}

impl Mul for I8OrI96 {
    type Output = I8OrI96;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        I8OrI96::mul(self, rhs)
    }
}

impl MulAssign for I8OrI96 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.mul_assign(&rhs)
    }
}

impl Sub for I8OrI96 {
    type Output = I8OrI96;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        I8OrI96::sub(self, rhs)
    }
}

impl SubAssign for I8OrI96 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs)
    }
}

impl Ord for I8OrI96 {
    #[inline]
    fn cmp(&self, other: &Self) -> ark_std::cmp::Ordering {
        use ark_std::cmp::Ordering;

        // Fast path for when both values are small.
        if self.is_small && other.is_small {
            return self.small_i8.cmp(&other.small_i8);
        }

        // Deconstruct into (hi, lo) parts to perform a 96-bit two's complement comparison.
        // If a value is small, we convert it to its large representation on the fly.
        let (self_hi, self_lo) = if self.is_small {
            let val = self.small_i8 as i128;
            ((val >> 64) as i32, val as u64)
        } else {
            (self.large_hi, self.large_lo)
        };

        let (other_hi, other_lo) = if other.is_small {
            let val = other.small_i8 as i128;
            ((val >> 64) as i32, val as u64)
        } else {
            (other.large_hi, other.large_lo)
        };

        // Compare the high parts first. If they differ, that determines the order.
        match self_hi.cmp(&other_hi) {
            Ordering::Equal => {
                // If high parts are the same, the order is determined by the low parts.
                self_lo.cmp(&other_lo)
            },
            order => order,
        }
    }
}

impl PartialOrd for I8OrI96 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<ark_std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl From<bool> for I8OrI96 {
    #[inline]
    fn from(value: bool) -> Self {
        if value {
            I8OrI96::one()
        } else {
            I8OrI96::zero()
        }
    }
}

impl From<i8> for I8OrI96 {
    #[inline]
    fn from(value: i8) -> Self {
        I8OrI96::from_i8(value)
    }
}

impl From<i16> for I8OrI96 {
    #[inline]
    fn from(value: i16) -> Self {
        I8OrI96::from_i128(value as i128)
    }
}

impl From<i32> for I8OrI96 {
    #[inline]
    fn from(value: i32) -> Self {
        I8OrI96::from_i128(value as i128)
    }
}

impl From<i64> for I8OrI96 {
    #[inline]
    fn from(value: i64) -> Self {
        I8OrI96::from_i128(value as i128)
    }
}

impl From<i128> for I8OrI96 {
    #[inline]
    fn from(value: i128) -> Self {
        I8OrI96::from_i128(value)
    }
}

impl From<isize> for I8OrI96 {
    #[inline]
    fn from(value: isize) -> Self {
        I8OrI96::from_i128(value as i128)
    }
}

impl From<u8> for I8OrI96 {
    #[inline]
    fn from(value: u8) -> Self {
        I8OrI96::from_i128(value as i128)
    }
}

impl From<u16> for I8OrI96 {
    #[inline]
    fn from(value: u16) -> Self {
        I8OrI96::from_i128(value as i128)
    }
}

impl From<u32> for I8OrI96 {
    #[inline]
    fn from(value: u32) -> Self {
        I8OrI96::from_i128(value as i128)
    }
}

impl From<u64> for I8OrI96 {
    #[inline]
    fn from(value: u64) -> Self {
        I8OrI96::from_i128(value as i128)
    }
}

impl From<usize> for I8OrI96 {
    #[inline]
    fn from(value: usize) -> Self {
        I8OrI96::from_i128(value as i128)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryFromU128Error;

impl core::fmt::Display for TryFromU128Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "u128 does not fit in signed 96-bit range")
    }
}

impl core::convert::TryFrom<u128> for I8OrI96 {
    type Error = TryFromU128Error;

    #[inline]
    fn try_from(value: u128) -> Result<Self, Self::Error> {
        // Signed 96-bit maximum is 2^95 - 1
        const I96_POS_MAX: u128 = (1u128 << 95) - 1;
        if value <= i8::MAX as u128 {
            Ok(I8OrI96::from_i8(value as i8))
        } else if value <= I96_POS_MAX {
            Ok(I8OrI96 {
                large_lo: value as u64,
                large_hi: (value >> 64) as i32,
                is_small: false,
                small_i8: 0,
            })
        } else {
            Err(TryFromU128Error)
        }
    }
}

impl Mul<S160> for I8OrI96 {
    type Output = S224;

    #[inline]
    fn mul(self, rhs: S160) -> Self::Output {
        // Determine sign of self
        let self_is_positive = if self.is_small {
            self.small_i8 >= 0
        } else {
            self.large_hi >= 0
        };

        // Extract rhs magnitude limbs
        let rhs_lo = rhs.magnitude_lo();
        let b0 = rhs_lo[0];
        let b1 = rhs_lo[1];
        let b2 = rhs.magnitude_hi() as u64; // widen for math
        let b1_is_zero = b1 == 0;
        let b2_is_zero = b2 == 0;
        let rhs_is_zero = (b0 | b1 | b2) == 0;

        if rhs_is_zero {
            return S224::zero();
        }

        // Compute absolute magnitude of self as 96-bit split (x0: u64, x1: u32)
        let (x0, x1_u32) = if self.is_small {
            let v = self.small_i8;
            let k = if v < 0 {
                (-(v as i16)) as u64
            } else {
                v as u64
            };
            (k, 0u32)
        } else {
            let hi = self.large_hi;
            if hi >= 0 {
                (self.large_lo, hi as u32)
            } else {
                // Two's-complement absolute: (~lo, ~hi) + 1
                let inv_lo = !self.large_lo;
                let inv_hi = !(hi as u32);
                let sum_lo = inv_lo.wrapping_add(1);
                let carry = (sum_lo == 0) as u32;
                let sum_hi = inv_hi.wrapping_add(carry);
                (sum_lo, sum_hi)
            }
        };

        // Compute magnitude product truncated to 224 bits: [r0,r1,r2] + hi32
        let (r0, r1, r2, hi32) = if x1_u32 == 0 {
            // Fast path: scalar (<= 8-bit) * 160-bit
            let k = x0;
            let mut c0 = 0u64;
            let r0 = mac_with_carry!(0u64, b0, k, &mut c0);

            if b1_is_zero {
                if b2_is_zero {
                    // Only 64-bit rhs
                    let r1 = c0;
                    (r0, r1, 0u64, 0u32)
                } else {
                    // 128-bit rhs via b2 only
                    let r1 = c0;
                    let mut hi = 0u64;
                    let r2 = mac_with_carry!(0u64, b2, k, &mut hi);
                    let hi32 = hi as u32;
                    (r0, r1, r2, hi32)
                }
            } else if b2_is_zero {
                // 128-bit rhs via b1 only
                let mut c1 = c0;
                let r1 = mac_with_carry!(0u64, b1, k, &mut c1);
                let r2 = c1;
                (r0, r1, r2, 0u32)
            } else {
                // Full 160-bit rhs
                let mut c1 = c0;
                let r1 = mac_with_carry!(0u64, b1, k, &mut c1);

                let mut c2 = 0u64;
                let mut r2 = mac_with_carry!(0u64, b2, k, &mut c2);
                r2 = adc!(r2, c1, &mut c2);
                let hi32 = c2 as u32;
                (r0, r1, r2, hi32)
            }
        } else {
            // General 96-bit (2 limbs: x0, x1_u32) times 160-bit (3 limbs: b0, b1, b2)
            let x1 = x1_u32 as u64;

            let mut c0 = 0u64;
            let r0 = mac_with_carry!(0u64, x0, b0, &mut c0);

            if b1_is_zero {
                if b2_is_zero {
                    // Only 64-bit rhs
                    let mut c1 = c0;
                    let r1 = mac_with_carry!(0u64, x1, b0, &mut c1);
                    let r2 = c1;
                    (r0, r1, r2, 0u32)
                } else {
                    // No b1, but have b2
                    let mut c1 = c0;
                    let r1 = mac_with_carry!(0u64, x1, b0, &mut c1);

                    let mut c2 = c1;
                    let r2 = mac_with_carry!(0u64, x0, b2, &mut c2);

                    let r3_low = ((c2 as u128)
                        + crate::biginteger::arithmetic::widening_mul(x1, b2)) as u64;
                    let hi32 = (r3_low & 0xFFFF_FFFF) as u32;
                    (r0, r1, r2, hi32)
                }
            } else if b2_is_zero {
                // No b2, but have b1
                let mut c1 = c0;
                let mut r1 = mac_with_carry!(0u64, x0, b1, &mut c1);
                r1 = mac_with_carry!(r1, x1, b0, &mut c1);

                let mut c2 = c1;
                let r2 = mac_with_carry!(0u64, x1, b1, &mut c2);
                let hi32 = c2 as u32;
                (r0, r1, r2, hi32)
            } else {
                // Full 160-bit rhs
                let mut c1 = c0;
                let mut r1 = mac_with_carry!(0u64, x0, b1, &mut c1);
                r1 = mac_with_carry!(r1, x1, b0, &mut c1);

                let mut c2 = c1;
                let mut r2 = mac_with_carry!(0u64, x0, b2, &mut c2);
                r2 = mac_with_carry!(r2, x1, b1, &mut c2);

                let r3_low = ((c2 as u128)
                    + crate::biginteger::arithmetic::widening_mul(x1, b2)) as u64;
                let hi32 = (r3_low & 0xFFFF_FFFF) as u32;
                (r0, r1, r2, hi32)
            }
        };

        // Combine sign; canonicalize zero to positive
        let mut is_positive = !(self_is_positive ^ rhs.is_positive());
        if (r0 | r1 | r2) == 0 && hi32 == 0 {
            is_positive = true;
        }

        S224::new([r0, r1, r2], hi32, is_positive)
    }
}

// ------------------------------------------------------------------------------------------------
// Canonical serialization
// ------------------------------------------------------------------------------------------------

impl CanonicalSerialize for I8OrI96 {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        mut w: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        // Print whether it is small (computed canonically from value), and the numeric value
        // as 96-bit two's complement represented by (hi: i32, lo: u64), derived from to_i128().
        let v = self.to_i128();
        let is_small_value = v >= i8::MIN as i128 && v <= i8::MAX as i128;
        let is_small_u8: u8 = if is_small_value { 1 } else { 0 };
        is_small_u8.serialize_with_mode(&mut w, compress)?;

        let hi: i32 = (v >> 64) as i32;
        let lo: u64 = v as u64;
        hi.serialize_with_mode(&mut w, compress)?;
        lo.serialize_with_mode(w, compress)
    }

    #[inline]
    fn serialized_size(&self, compress: Compress) -> usize {
        (self.is_small as u8).serialized_size(compress)
            + (0i32).serialized_size(compress)
            + (0u64).serialized_size(compress)
    }
}

impl CanonicalDeserialize for I8OrI96 {
    #[inline]
    fn deserialize_with_mode<R: Read>(
        mut r: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let _is_small = u8::deserialize_with_mode(&mut r, compress, validate)?;
        let hi = i32::deserialize_with_mode(&mut r, compress, validate)?;
        let lo = u64::deserialize_with_mode(r, compress, validate)?;
        let v: i128 = ((hi as i128) << 64) | (lo as i128);
        Ok(I8OrI96::from_i128(v))
    }
}

impl Valid for I8OrI96 {
    #[inline]
    fn check(&self) -> Result<(), SerializationError> {
        // All bit patterns of the struct represent either a small i8 or a 96-bit two's complement value.
        // No additional invariants beyond that; always valid.
        Ok(())
    }
}
