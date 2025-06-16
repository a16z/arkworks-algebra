# GLV4 Demo: 4-Dimensional GLV for BN254 G2

This is a standalone demonstration of the 4-dimensional GLV (Gallant-Lambert-Vanstone) decomposition implementation for the BN254 G2 elliptic curve group.

## What This Demonstrates

The 4D GLV optimization reduces scalar multiplication complexity by decomposing a ~256-bit scalar into four ~64-bit coefficients:

```
s = a0 + a1*λ_glv + a2*λ_glv² + b*λ_frob
```

Where:
- `λ_glv` is the GLV eigenvalue for BN254
- `λ_frob` is the Frobenius eigenvalue (field characteristic p)
- Each coefficient (a0, a1, a2, b) is significantly smaller than the original scalar

## Implementation

This implementation:
1. **Extends arkworks-algebra** with a new `GLV4Config` trait
2. **Implements the trait** for BN254 G2 using precomputed lattice coefficients
3. **Follows the SageMath approach** from the `4dim.py` script
4. **Combines two endomorphisms**: GLV (ψ) and Frobenius (φ)

## Running the Demo

```bash
cd glv4-demo
cargo run
```

The demo will:
- Generate a random scalar
- Perform 4D decomposition
- Verify algebraically that the decomposition is correct
- Test scalar multiplication on the curve using three methods:
  - Standard scalar multiplication
  - 4D GLV optimized multiplication  
  - Manual computation using the decomposition
- Show the efficiency gains achieved

## Expected Output

The demo shows:
- Bit length reduction (typically ~4x improvement)
- Algebraic verification of the decomposition
- Curve point verification that all methods produce the same result
- The explicit decomposition formula

## Technical Details

- **Curve**: BN254 G2 (pairing-friendly curve)
- **Method**: Lattice-based scalar decomposition using CVP approximation
- **Endomorphisms**: GLV (complex multiplication) + Frobenius (field automorphism)
- **Optimization**: Reduces ~256-bit scalar to four ~64-bit scalars
- **Integration**: Clean extension of existing arkworks GLV infrastructure