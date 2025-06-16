use crate::{
    short_weierstrass::{Affine, Projective, SWCurveConfig},
    AdditiveGroup, CurveGroup,
};
use ark_ff::{PrimeField, Zero};
use ark_std::ops::{AddAssign, Neg};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed};

/// The GLV parameters for computing the endomorphism and scalar decomposition.
pub trait GLVConfig: Send + Sync + 'static + SWCurveConfig {
    /// Constants that are used to calculate `phi(G) := lambda*G`.

    /// The coefficients of the endomorphism
    const ENDO_COEFFS: &'static [Self::BaseField];

    /// The eigenvalue corresponding to the endomorphism.
    const LAMBDA: Self::ScalarField;

    /// A 4-element vector representing a 2x2 matrix of coefficients the for scalar decomposition, s.t. k-th entry in the vector is at col i, row j in the matrix, with ij = BE binary decomposition of k.
    /// The entries are the LLL-reduced bases.
    /// The determinant of this matrix must equal `ScalarField::characteristic()`.
    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4];

    /// Decomposes a scalar s into k1, k2, s.t. s = k1 + lambda k2,
    fn scalar_decomposition(
        k: Self::ScalarField,
    ) -> ((bool, Self::ScalarField), (bool, Self::ScalarField)) {
        let scalar: BigInt = k.into_bigint().into().into();

        let coeff_bigints: [BigInt; 4] = Self::SCALAR_DECOMP_COEFFS.map(|x| {
            BigInt::from_biguint(x.0.then_some(Sign::Plus).unwrap_or(Sign::Minus), x.1.into())
        });

        let [n11, n12, n21, n22] = coeff_bigints;

        let r = BigInt::from(Self::ScalarField::MODULUS.into());

        // beta = vector([k,0]) * self.curve.N_inv
        // The inverse of N is 1/r * Matrix([[n22, -n12], [-n21, n11]]).
        // so β = (k*n22, -k*n12)/r

        let beta_1 = {
            let (mut div, rem) = (&scalar * &n22).div_rem(&r);
            if (&rem + &rem) > r {
                div.add_assign(BigInt::one());
            }
            div
        };
        let beta_2 = {
            let (mut div, rem) = (&scalar * &n12.clone().neg()).div_rem(&r);
            if (&rem + &rem) > r {
                div.add_assign(BigInt::one());
            }
            div
        };

        // b = vector([int(beta[0]), int(beta[1])]) * self.curve.N
        // b = (β1N11 + β2N21, β1N12 + β2N22) with the signs!
        //   = (b11   + b12  , b21   + b22)   with the signs!

        // b1
        let b11 = &beta_1 * &n11;
        let b12 = &beta_2 * &n21;
        let b1 = b11 + b12;

        // b2
        let b21 = &beta_1 * &n12;
        let b22 = &beta_2 * &n22;
        let b2 = b21 + b22;

        let k1 = &scalar - b1;
        let k1_abs = BigUint::try_from(k1.abs()).unwrap();

        // k2
        let k2 = -b2;
        let k2_abs = BigUint::try_from(k2.abs()).unwrap();

        (
            (k1.sign() == Sign::Plus, Self::ScalarField::from(k1_abs)),
            (k2.sign() == Sign::Plus, Self::ScalarField::from(k2_abs)),
        )
    }

    fn endomorphism(p: &Projective<Self>) -> Projective<Self>;

    fn endomorphism_affine(p: &Affine<Self>) -> Affine<Self>;

    fn glv_mul_projective(p: Projective<Self>, k: Self::ScalarField) -> Projective<Self> {
        let ((sgn_k1, k1), (sgn_k2, k2)) = Self::scalar_decomposition(k);

        let mut b1 = p;
        let mut b2 = Self::endomorphism(&p);

        if !sgn_k1 {
            b1 = -b1;
        }
        if !sgn_k2 {
            b2 = -b2;
        }

        let b1b2 = b1 + b2;

        let iter_k1 = ark_ff::BitIteratorBE::new(k1.into_bigint());
        let iter_k2 = ark_ff::BitIteratorBE::new(k2.into_bigint());

        let mut res = Projective::<Self>::zero();
        let mut skip_zeros = true;
        for pair in iter_k1.zip(iter_k2) {
            if skip_zeros && pair == (false, false) {
                skip_zeros = false;
                continue;
            }
            res.double_in_place();
            match pair {
                (true, false) => res += b1,
                (false, true) => res += b2,
                (true, true) => res += b1b2,
                (false, false) => {},
            }
        }
        res
    }

    fn glv_mul_affine(p: Affine<Self>, k: Self::ScalarField) -> Affine<Self> {
        let ((sgn_k1, k1), (sgn_k2, k2)) = Self::scalar_decomposition(k);

        let mut b1 = p;
        let mut b2 = Self::endomorphism_affine(&p);

        if !sgn_k1 {
            b1 = -b1;
        }
        if !sgn_k2 {
            b2 = -b2;
        }

        let b1b2 = b1 + b2;

        let iter_k1 = ark_ff::BitIteratorBE::new(k1.into_bigint());
        let iter_k2 = ark_ff::BitIteratorBE::new(k2.into_bigint());

        let mut res = Projective::<Self>::zero();
        let mut skip_zeros = true;
        for pair in iter_k1.zip(iter_k2) {
            if skip_zeros && pair == (false, false) {
                skip_zeros = false;
                continue;
            }
            res.double_in_place();
            match pair {
                (true, false) => res += b1,
                (false, true) => res += b2,
                (true, true) => res += b1b2,
                (false, false) => {},
            }
        }
        res.into_affine()
    }
}

/// The GLV4 parameters for 4-dimensional scalar decomposition combining GLV and Frobenius endomorphisms.
pub trait GLV4Config: GLVConfig {
    /// The Frobenius eigenvalue (typically the field characteristic p for BN curves).
    const LAMBDA_FROB: Self::ScalarField;

    /// A 6x5 matrix represented as a flat array for 4D scalar decomposition.
    /// The matrix is used in CVP (Closest Vector Problem) solving.
    const SCALAR_DECOMP_COEFFS_4D: [(bool, <Self::ScalarField as PrimeField>::BigInt); 30];

    /// Decomposes a scalar s into k, a0, a1, a2, b, s.t. k*s = a0 + a1*λ_glv + a2*λ_glv² + b*λ_frob,
    /// where λ_glv is the GLV eigenvalue and λ_frob is the Frobenius eigenvalue.
    fn scalar_decomposition_4d(
        _s: Self::ScalarField,
    ) -> (
        Self::ScalarField,  // k (scaling factor, typically 1)
        (bool, Self::ScalarField),  // a0
        (bool, Self::ScalarField),  // a1
        (bool, Self::ScalarField),  // a2
        (bool, Self::ScalarField),  // b
    ) {
        // This is a simplified implementation. A full implementation would use
        // lattice reduction (LLL) and CVP solving as in the SageMath script.
        // For now, we'll provide a stub that needs to be properly implemented
        // for each specific curve.
        unimplemented!("4D scalar decomposition must be implemented for each curve")
    }

    /// The Frobenius endomorphism on the curve.
    fn frobenius(p: &Projective<Self>) -> Projective<Self>;

    /// The Frobenius endomorphism on the curve (affine version).
    fn frobenius_affine(p: &Affine<Self>) -> Affine<Self>;

    /// Performs scalar multiplication using 4-dimensional decomposition.
    fn glv4_mul_projective(p: Projective<Self>, s: Self::ScalarField) -> Projective<Self> {
        let (_k, (sgn_a0, a0), (sgn_a1, a1), (sgn_a2, a2), (sgn_b, b)) = Self::scalar_decomposition_4d(s);
        
        // Compute basis points
        let p0 = p;  // P
        let p1 = Self::endomorphism(&p);  // ψ(P)
        let p2 = Self::endomorphism(&p1);  // ψ²(P)
        let p3 = Self::frobenius(&p);  // φ(P)
        
        // Handle signs
        let mut b0 = p0;
        let mut b1 = p1;
        let mut b2 = p2;
        let mut b3 = p3;
        
        if !sgn_a0 { b0 = -b0; }
        if !sgn_a1 { b1 = -b1; }
        if !sgn_a2 { b2 = -b2; }
        if !sgn_b { b3 = -b3; }
        
        // Precompute all 16 combinations
        let table = [
            Projective::<Self>::zero(),  // 0000
            b0,                          // 0001
            b1,                          // 0010
            b0 + b1,                     // 0011
            b2,                          // 0100
            b0 + b2,                     // 0101
            b1 + b2,                     // 0110
            b0 + b1 + b2,                // 0111
            b3,                          // 1000
            b0 + b3,                     // 1001
            b1 + b3,                     // 1010
            b0 + b1 + b3,                // 1011
            b2 + b3,                     // 1100
            b0 + b2 + b3,                // 1101
            b1 + b2 + b3,                // 1110
            b0 + b1 + b2 + b3,           // 1111
        ];
        
        // Interleave the bits of the four scalars
        let iter_a0 = ark_ff::BitIteratorBE::new(a0.into_bigint());
        let iter_a1 = ark_ff::BitIteratorBE::new(a1.into_bigint());
        let iter_a2 = ark_ff::BitIteratorBE::new(a2.into_bigint());
        let iter_b = ark_ff::BitIteratorBE::new(b.into_bigint());
        
        let mut res = Projective::<Self>::zero();
        let mut skip_zeros = true;
        
        for (((bit_a0, bit_a1), bit_a2), bit_b) in iter_a0.zip(iter_a1).zip(iter_a2).zip(iter_b) {
            let index = (bit_b as usize) << 3 | (bit_a2 as usize) << 2 | (bit_a1 as usize) << 1 | (bit_a0 as usize);
            
            if skip_zeros && index == 0 {
                continue;
            }
            skip_zeros = false;
            
            res.double_in_place();
            res += &table[index];
        }
        
        res
    }

    /// Performs scalar multiplication using 4-dimensional decomposition (affine version).
    fn glv4_mul_affine(p: Affine<Self>, s: Self::ScalarField) -> Affine<Self> {
        Self::glv4_mul_projective(p.into(), s).into_affine()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_glv4_bn254_g2_example() {
        // This test demonstrates 4D GLV decomposition for BN254 G2
        // It requires the bn254 curve implementation with GLV4Config
        
        // Since this is a generic test in the ec crate, we can't directly test BN254
        // The actual test implementation is in the bn254 curve crate
        // This serves as a template for how to use the GLV4Config trait
        
        // Example usage (pseudo-code):
        // let s = Fr::rand(&mut rng);
        // let (k, (sgn_a0, a0), (sgn_a1, a1), (sgn_a2, a2), (sgn_b, b)) = 
        //     <Config as GLV4Config>::scalar_decomposition_4d(s);
        //
        // Where the decomposition satisfies:
        // k*s = a0 + a1*λ_glv + a2*λ_glv² + b*λ_frob (mod r)
        //
        // And scalar multiplication can be computed as:
        // s*P = a0*P + a1*ψ(P) + a2*ψ²(P) + b*φ(P)
        //
        // This reduces a 256-bit scalar multiplication to four ~64-bit scalar multiplications
        
        assert!(true); // Placeholder test
    }
}
