/// Right-hand instruction input value used by zkVM instruction logic.
///
/// Captures the semantic signedness of the right operand at XLEN width.
/// - `Unsigned(u64)`: operand is interpreted as an XLEN-bit unsigned word
/// - `Signed(i64)`: operand is interpreted as an XLEN-bit two's-complement signed word
///
/// Helper methods provide width-aware projections to `u64`/`i64` and a
/// canonical unsigned representation for lookup key construction.
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct U64AndSign {
    pub magnitude: u64,
    pub is_negative: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct U128AndSign {
    pub magnitude: u128,
    pub is_negative: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct U160AndSIgn {
    pub magnitude_low: u128,
    pub magnitude_high: u32,
    pub is_negative: bool,
}


impl U64AndSign {
    /// Construct a nonnegative value from an unsigned magnitude.
    #[inline]
    #[allow(non_snake_case)]
    pub fn Unsigned(u: u64) -> Self {
        Self {
            magnitude: u,
            is_negative: false,
        }
    }

    /// Construct a value from a signed 64-bit word.
    /// Negative zero is normalized to +0.
    #[inline]
    #[allow(non_snake_case)]
    pub fn Signed(s: i64) -> Self {
        if s >= 0 {
            Self {
                magnitude: s as u64,
                is_negative: false,
            }
        } else {
            let mag = (-(s as i128)) as u64;
            Self {
                magnitude: mag,
                is_negative: mag != 0,
            }
        }
    }

    /// Return the value as an unsigned 64-bit word (XLEN=64 view).
    #[inline]
    pub fn as_u64(&self) -> u64 {
        self.magnitude
    }

    /// Return the value as an unsigned 32-bit word (XLEN=32 view).
    #[inline]
    pub fn as_u32(&self) -> u32 {
        self.magnitude as u32
    }

    /// Return the value as an unsigned 8-bit word (XLEN=8 view).
    #[inline]
    pub fn as_u8(&self) -> u8 {
        self.magnitude as u8
    }

    /// Return the value as a signed 64-bit word (XLEN=64 view).
    #[inline]
    pub fn as_i64(&self) -> i64 {
        self.as_i128() as i64
    }

    /// Return the value as a signed 32-bit word (XLEN=32 view).
    /// This is a truncating conversion.
    #[inline]
    pub fn as_i32(&self) -> i32 {
        self.as_i128() as i32
    }

    /// Return the value as a signed 8-bit word (XLEN=8 view).
    /// This is a truncating conversion.
    #[inline]
    pub fn as_i8(&self) -> i8 {
        self.as_i128() as i8
    }

    /// Return the value widened to i128.
    #[inline]
    pub fn as_i128(&self) -> i128 {
        if self.is_negative && self.magnitude != 0 {
            -(self.magnitude as i128)
        } else {
            self.magnitude as i128
        }
    }

    /// Return a canonical unsigned representation suitable for lookup keys.
    ///
    /// This is the XLEN-masked view of the value, promoted to `u128`.
    #[inline]
    pub fn to_u128_lookup<const XLEN: usize>(&self) -> u128 {
        match XLEN {
            64 => self.as_u64() as u128,
            32 => self.as_u32() as u128,
            8 => self.as_u8() as u128,
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    /// Returns true if the value is negative.
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.is_negative && self.magnitude != 0
    }

    /// Returns true if the value is nonnegative (>= 0).
    #[inline]
    pub fn is_positive(&self) -> bool {
        !self.is_negative()
    }
}

impl U128AndSign {
    /// Construct a nonnegative value from an unsigned magnitude.
    #[inline]
    #[allow(non_snake_case)]
    pub fn Unsigned(u: u128) -> Self {
        Self {
            magnitude: u,
            is_negative: false,
        }
    }

    /// Construct a value from a signed 128-bit word.
    /// Negative zero is normalized to +0.
    #[inline]
    #[allow(non_snake_case)]
    pub fn Signed(s: i128) -> Self {
        if s >= 0 {
            Self {
                magnitude: s as u128,
                is_negative: false,
            }
        } else {
            // Handle i128::MIN without overflow
            let mag = if s == i128::MIN {
                1u128 << 127
            } else {
                (-s) as u128
            };
            Self {
                magnitude: mag,
                is_negative: mag != 0,
            }
        }
    }

    #[inline]
    pub fn as_u128(&self) -> u128 {
        self.magnitude
    }

    #[inline]
    pub fn as_i128(&self) -> i128 {
        if self.is_negative && self.magnitude != 0 {
            -(self.magnitude as i128)
        } else {
            self.magnitude as i128
        }
    }

    /// Returns true if the value is negative.
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.is_negative && self.magnitude != 0
    }

    /// Returns true if the value is nonnegative (>= 0).
    #[inline]
    pub fn is_positive(&self) -> bool {
        !self.is_negative()
    }
}

impl core::cmp::PartialOrd for U64AndSign {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl core::cmp::Ord for U64AndSign {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        let a = self.as_i128();
        let b = other.as_i128();
        a.cmp(&b)
    }
}

impl core::cmp::PartialOrd for U128AndSign {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl core::cmp::Ord for U128AndSign {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        let a = self.as_i128();
        let b = other.as_i128();
        a.cmp(&b)
    }
}

impl Default for U64AndSign {
    fn default() -> Self {
        U64AndSign::Unsigned(0)
    }
}

impl Default for U128AndSign {
    fn default() -> Self {
        U128AndSign::Unsigned(0)
    }
}

impl Valid for U64AndSign {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl Valid for U128AndSign {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for U64AndSign {
    fn serialize_with_mode<W: ark_std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        if self.is_negative() {
            1u8.serialize_with_mode(&mut writer, compress)?;
            let s = self.as_i128() as i64;
            s.serialize_with_mode(writer, compress)
        } else {
            0u8.serialize_with_mode(&mut writer, compress)?;
            let u = self.magnitude;
            u.serialize_with_mode(writer, compress)
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        0u8.serialized_size(compress)
            + if self.is_negative() {
                (self.as_i128() as i64).serialized_size(compress)
            } else {
                self.magnitude.serialized_size(compress)
            }
    }
}

impl CanonicalDeserialize for U64AndSign {
    fn deserialize_with_mode<R: ark_std::io::Read>(
        mut reader: R,
        compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        match tag {
            0 => {
                let u = u64::deserialize_with_mode(reader, compress, Validate::No)?;
                Ok(U64AndSign::Unsigned(u))
            }
            1 => {
                let s = i64::deserialize_with_mode(reader, compress, Validate::No)?;
                Ok(U64AndSign::Signed(s))
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl CanonicalSerialize for U128AndSign {
    fn serialize_with_mode<W: ark_std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        if self.is_negative() {
            1u8.serialize_with_mode(&mut writer, compress)?;
            let s = self.as_i128();
            s.serialize_with_mode(writer, compress)
        } else {
            0u8.serialize_with_mode(&mut writer, compress)?;
            let u = self.magnitude;
            u.serialize_with_mode(writer, compress)
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        0u8.serialized_size(compress)
            + if self.is_negative() {
                self.as_i128().serialized_size(compress)
            } else {
                self.magnitude.serialized_size(compress)
            }
    }
}

impl CanonicalDeserialize for U128AndSign {
    fn deserialize_with_mode<R: ark_std::io::Read>(
        mut reader: R,
        compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        match tag {
            0 => {
                let u = u128::deserialize_with_mode(reader, compress, Validate::No)?;
                Ok(U128AndSign::Unsigned(u))
            }
            1 => {
                let s = i128::deserialize_with_mode(reader, compress, Validate::No)?;
                Ok(U128AndSign::Signed(s))
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}

#[cfg(test)]
/// Validate that specialized projections as_{u,i}{8,32,64} are equivalent to
/// widening to i128 and narrowing back for both Unsigned and Signed variants
/// under XLEN views 8, 32, and 64.
mod tests {
    use super::U64AndSign as RIV;
    use rand::Rng;

    fn check_equivalence(v: RIV) {
        let i128_wide = v.as_i128();

        // XLEN=8
        assert_eq!(i128_wide as u8, v.as_u8());
        assert_eq!(i128_wide as i8, v.as_i8());

        // XLEN=32
        assert_eq!(i128_wide as u32, v.as_u32());
        assert_eq!(i128_wide as i32, v.as_i32());

        // XLEN=64
        assert_eq!(i128_wide as u64, v.as_u64());
        assert_eq!(i128_wide as i64, v.as_i64());
    }

    #[test]
    fn projections_match_i128_path_unsigned() {
        let cases: &[u64] = &[
            0,
            1,
            0x7F,
            0x80,
            0xFF,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_FFFF,
            0x7FFF_FFFF_FFFF_FFFF,
            0x8000_0000_0000_0000,
            0xFFFF_FFFF_FFFF_FFFF,
        ];
        for &u in cases {
            check_equivalence(RIV::Unsigned(u));
        }

        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            check_equivalence(RIV::Unsigned(rng.gen()));
        }
    }

    #[test]
    fn projections_match_i128_path_signed() {
        let cases: &[i64] = &[
            0,
            1,
            -1,
            i64::MIN,
            i64::MAX,
            -128,
            127,
            -0x8000_0000,
            0x7FFF_FFFF,
        ];
        for &s in cases {
            check_equivalence(RIV::Signed(s));
        }

        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            check_equivalence(RIV::Signed(rng.gen()));
        }
    }
}


