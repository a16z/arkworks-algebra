"""
BN254 G2 Scalar Decomposition using Endomorphisms - Corrected Version

This script demonstrates how to decompose a scalar multiplication on BN254's G2 curve
using both the cube root of unity and Frobenius endomorphisms.

The key correction is the proper implementation of the Frobenius endomorphism
for G2, which must account for the sextic twist structure.
"""

def solve_cvp(B, t, verbose=False):
    """
    Approximately and efficiently solves the closest vector problem.
    """
    t_ = t - B.stack(t).gram_schmidt()[0].row(-1)
    if verbose:
        print("Target vector projection:")
        print(numerical_approx(t_, digits=4))
    
    B_ = B.LLL()
    if verbose:
        print("\nLLL-reduced basis:")
        print(numerical_approx(B_, digits=4))

    c = B_.solve_left(t_)
    c_ = vector(map(round, c))
    if verbose:
        print("\nRound-off errors:")
        print(numerical_approx(vector(map(abs, c - c_)), digits=4))
    
    return c_ * B_

def solve_cvp2(B, t, scale_factors=None, verbose=False):
    """
    A wrapper of `solve_cvp` to perform coordinate scaling.
    """
    if not scale_factors:
        scale_factors = [1] * B.ncols()
    
    if verbose:
        print("Scale factors:")
        print(numerical_approx(vector(scale_factors), digits=4), '\n')
    
    scale_matrix = diagonal_matrix(scale_factors)
    return solve_cvp(B*scale_matrix, t*scale_matrix, verbose) * scale_matrix^-1

# BN254 parameters
x = 4965661367192848881
p = 36*x^4 + 36*x^3 + 24*x^2 + 6*x + 1
r = 36*x^4 + 36*x^3 + 18*x^2 + 6*x + 1

Fr = GF(r)
Fp = GF(p)

print("=== BN254 Parameters ===")
print(f"x = {x}")
print(f"p = {p} ({p.nbits()} bits)")
print(f"r = {r} ({r.nbits()} bits)")

# Define Fp^2 for G2
R.<w> = PolynomialRing(Fp)
Fp2.<u> = Fp.extension(w^2 + 1)

# Helper function to create Fp2 elements
def fp2(a, b=0):
    """Create an Fp2 element from real and imaginary parts"""
    return Fp(a) + Fp(b) * u

# Curve parameters from the Rust code
# G1: y² = x³ + 3
# G2 uses sextic twist with B = 3/(u+9)

# First, let's define xi = u + 9 (the twist parameter)
xi = u + 9

# The twist curve coefficient is 3/xi
twist_B = 3 / xi

# Define curves
E1 = EllipticCurve(Fp, [0, 3])
E2 = EllipticCurve(Fp2, [0, twist_B])

print(f"\nG1 curve: y² = x³ + 3")
print(f"G2 curve: y² = x³ + {twist_B}")

# G2 generator from the Rust code
g2_x = fp2(10857046999023057135944570762232829481370756359578518086990519993285655852781,
           11559732032986387107991004021392285783925812861821192530917403151452391805634)
g2_y = fp2(8495653923123431417604973247489272438418190587263600148770280649306958101930,
           4082367875863433681332203403145435568316851327593401208105741076214120093531)

G2 = E2(g2_x, g2_y)
print(f"G2 generator order check...")
assert r * G2 == E2(0), "G2 generator does not have order r"
print("✓ G2 generator has order r")

# Compute endomorphism eigenvalues
# Cube root of unity in Fr
zeta3 = Fr.primitive_element()^((r-1)//3)
assert zeta3^3 == 1 and zeta3 != 1

# For the Frobenius on G2, we need to find the correct eigenvalue
# The trace of Frobenius on E(Fp^2) is related to the trace on E(Fp)
t1 = p + 1 - E1.order()  # trace on E(Fp)
print(f"\nTrace on E(Fp): {t1}")

# For degree 2 extension, the eigenvalues satisfy:
# λ^2 - t2*λ + p^2 = 0
# where t2 is the trace on E(Fp^2)

# For G2 (the r-torsion subgroup of E'(Fp^2)), we need the eigenvalue mod r
# For BN curves, this works out to be q mod r where q = p
lambda_frob = Fr(p)

print(f"\n=== Endomorphism Eigenvalues ===")
print(f"ζ₃ (cube root) = {zeta3}")
print(f"λ (Frobenius) = {lambda_frob}")

# Frobenius coefficients from the Rust code
# PSI_X = (u+9)^((p-1)/3)
PSI_X = fp2(21575463638280843010398324269430826099269044274347216827212613867836435027261,
            10307601595873709700152284273816112264069230130616436755625194854815875713954)

# PSI_Y = (u+9)^((p-1)/2)
PSI_Y = fp2(2821565182194536844548159561693502659359617185244120367078079554186484126554,
            3505843767911556378687030309984248845540243509899259641013678093033130930403)

print(f"\nFrobenius coefficients:")
print(f"PSI_X = xi^((p-1)/3)")
print(f"PSI_Y = xi^((p-1)/2)")

# Verify these are correct
assert PSI_X == xi^((p-1)//3), "PSI_X mismatch"
assert PSI_Y == xi^((p-1)//2), "PSI_Y mismatch"
print("✓ Frobenius coefficients verified")

# GLV endomorphism coefficient from Rust code
# This value needs to be checked - it might be a cube root of unity in Fp2
glv_coeff_value = 21888242871839275220042445260109153167277707414472061641714758635765020556616
GLV_ENDO_COEFF = fp2(glv_coeff_value, 0)

# The GLV lambda from the Rust code (LAMBDA value)
glv_lambda = Fr(4407920970296243842393367215006156084916469457145843978461)

# Let's check what this value is
print(f"\nGLV coefficient value: {glv_coeff_value}")
print(f"p = {p}")
print(f"p - 1 = {p - 1}")
print(f"glv_coeff_value - p = {glv_coeff_value - p}")

# Check if it's -1 in a different field
# The coefficient might be from a different prime field
# Let's check if it's related to the ark-bn254 prime
ark_p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
print(f"\nark-bn254 p = {ark_p}")
print(f"glv_coeff_value - ark_p = {glv_coeff_value - ark_p}")
print(f"Is glv_coeff_value = ark_p - 1? {glv_coeff_value == ark_p - 1}")

if glv_coeff_value == ark_p - 1:
    print("\nThe GLV coefficient is -1 in the ark-bn254 field")
    # Update our field definitions to use ark-bn254 parameters
    p = ark_p
    Fp = GF(p)
    R.<w> = PolynomialRing(Fp)
    Fp2.<u> = Fp.extension(w^2 + 1)
    
    # Redefine fp2 helper
    def fp2(a, b=0):
        """Create an Fp2 element from real and imaginary parts"""
        return Fp(a) + Fp(b) * u
    
    # Recreate the curve and points with the correct field
    xi = u + 9
    twist_B = 3 / xi
    E2 = EllipticCurve(Fp2, [0, twist_B])
    
    # Recreate G2 generator
    g2_x = fp2(10857046999023057135944570762232829481370756359578518086990519993285655852781,
               11559732032986387107991004021392285783925812861821192530917403151452391805634)
    g2_y = fp2(8495653923123431417604973247489272438418190587263600148770280649306958101930,
               4082367875863433681332203403145435568316851327593401208105741076214120093531)
    G2 = E2(g2_x, g2_y)
    
    # Recreate Frobenius coefficients
    PSI_X = fp2(21575463638280843010398324269430826099269044274347216827212613867836435027261,
                10307601595873709700152284273816112264069230130616436755625194854815875713954)
    PSI_Y = fp2(2821565182194536844548159561693502659359617185244120367078079554186484126554,
                3505843767911556378687030309984248845540243509899259641013678093033130930403)
    
    # Now GLV_ENDO_COEFF is -1 in Fp2
    GLV_ENDO_COEFF = fp2(ark_p - 1, 0)  # This is -1 in Fp2

# Find the actual cube roots of unity in Fp2
# They satisfy x^3 = 1, so x^3 - 1 = 0
P_poly.<x> = PolynomialRing(Fp2)
cube_roots = (x^3 - 1).roots()
cube_roots_list = [r[0] for r in cube_roots]
print(f"\nCube roots of unity in Fp2: {cube_roots_list}")

# Check if GLV_ENDO_COEFF is among them
omega_glv = None
if GLV_ENDO_COEFF in cube_roots_list:
    omega_glv = GLV_ENDO_COEFF
    print(f"✓ GLV_ENDO_COEFF is a cube root of unity")
else:
    print(f"GLV_ENDO_COEFF is not a cube root of unity, searching...")
    # Find the non-trivial cube root that gives the right eigenvalue
    for root in cube_roots_list:
        if root != 1:
            # Check if this gives the right eigenvalue
            test_pt = 5 * G2
            try:
                psi_test = E2(root * test_pt[0], test_pt[1])
                if psi_test == ZZ(glv_lambda) * test_pt:
                    omega_glv = root
                    print(f"Found correct cube root: {omega_glv}")
                    break
            except:
                pass
    
    if omega_glv is None:
        print("Warning: Could not find correct cube root for GLV endomorphism")
        # Default to a non-trivial cube root
        omega_glv = [r for r in cube_roots_list if r != 1][0]
        print(f"Using cube root: {omega_glv}")

print(f"\nSelected omega for GLV: {omega_glv}")
print(f"Verification: omega^3 = {omega_glv^3}")

# Define the corrected endomorphisms
def psi(point):
    """Cube root endomorphism on G2 (GLV endomorphism)"""
    if point.is_zero():
        return point
    return E2(omega_glv * point[0], point[1])

def psi2(point):
    """Square of cube root endomorphism"""
    if point.is_zero():
        return point
    return E2(omega_glv^2 * point[0], point[1])

def frobenius_g2(point):
    """Frobenius endomorphism on G2 (untwist-Frobenius-twist)"""
    if point.is_zero():
        return point
    
    # Apply Frobenius map to coordinates
    x_frob = point[0]^p
    y_frob = point[1]^p
    
    # Apply twist correction
    x_result = x_frob * PSI_X
    y_result = y_frob * PSI_Y
    
    return E2(x_result, y_result)

# Test the endomorphisms
print(f"\n=== Testing Endomorphisms ===")

# Test psi (cube root endomorphism)
test_point = 17 * G2  # Use a small multiple for testing
psi_test = psi(test_point)

print(f"Testing ψ endomorphism...")
print(f"ψ acts with eigenvalue λ_glv = {glv_lambda}")
psi_eigen_test = ZZ(glv_lambda) * test_point
if psi_test == psi_eigen_test:
    print("✓ ψ endomorphism verified")
else:
    print("✗ ψ endomorphism failed - checking alternative eigenvalues")
    # Try other possibilities
    if psi_test == ZZ(glv_lambda^2) * test_point:
        print(f"ψ acts with eigenvalue λ_glv² = {glv_lambda^2}")
        glv_lambda = glv_lambda^2
    else:
        print("Could not determine ψ eigenvalue")

# Test Frobenius
print(f"\nTesting Frobenius endomorphism...")
frob_test = frobenius_g2(test_point)
frob_eigen_test = ZZ(lambda_frob) * test_point
if frob_test == frob_eigen_test:
    print("✓ Frobenius endomorphism verified")
else:
    print("✗ Frobenius endomorphism failed")
    # Try alternative eigenvalues
    for alt in [Fr(x), Fr(x+1), Fr(p^2 % r)]:
        if frob_test == ZZ(alt) * test_point:
            print(f"Frobenius acts with eigenvalue {alt}")
            lambda_frob = alt
            break

# Additional: demonstrate the Frobenius endomorphism works
print(f"\n=== Frobenius Endomorphism Demo ===")
# For any point P, we have φ(P) = p*P in the r-torsion
test_p = 123 * G2
frob_p = frobenius_g2(test_p)
p_times_p = ZZ(lambda_frob) * test_p
print(f"φ(P) = p·P (mod r): {'✓' if frob_p == p_times_p else '✗'}")

print(f"\n=== Full 6-Dimensional Decomposition ===")

# Configuration - Set your desired scaling bits here
SCALE_BITS = {
    'k': 128,   # Keep k close to 1
    'a0': 40,   # Coefficients for identity
    'a1': 40,   # Coefficients for ζ₃
    'a2': 40,   # Coefficients for ζ₃²
    'b0': 40,   # Coefficients for λ
    'b1': 40,   # Coefficients for λ·ζ₃
    'b2': 40    # Coefficients for λ·ζ₃²
}

# Generate a new random scalar for the 6D decomposition
s_random_6d = Fr.random_element()
s_bits_6d = ZZ(s_random_6d).nbits()
print(f"\nNew random scalar: {s_bits_6d} bits")
print(f"Using scaling bits: {SCALE_BITS}")

# For the 6D decomposition, we need both eigenvalues
# glv_lambda for the cube root endomorphism
# lambda_frob for the Frobenius endomorphism
print(f"\nEigenvalues:")
print(f"  GLV (cube root): {glv_lambda}")
print(f"  Frobenius: {lambda_frob}")

# Build the lattice matrix
L = Matrix(QQ, [
    [QQ(s_random_6d)],
    [r],
    [-1],
    [-QQ(glv_lambda)],
    [-QQ(glv_lambda^2)],
    [-QQ(lambda_frob)],
    [-QQ(lambda_frob * glv_lambda)],
    [-QQ(lambda_frob * glv_lambda^2)]
])

# Right part: scaled identity matrix for the 7 unknowns
R = Matrix(QQ, 8, 7)
R[1,0] = 1/(2^SCALE_BITS['k'])
R[2,1] = -1/(2^SCALE_BITS['a0'])
R[3,2] = -1/(2^SCALE_BITS['a1'])
R[4,3] = -1/(2^SCALE_BITS['a2'])
R[5,4] = -1/(2^SCALE_BITS['b0'])
R[6,5] = -1/(2^SCALE_BITS['b1'])
R[7,6] = -1/(2^SCALE_BITS['b2'])

# Combine into the full lattice matrix
M = block_matrix([[L, R]])

print(f"\nLattice matrix dimensions: {M.nrows()}x{M.ncols()}")

# Target vector and relevance weights
t = vector([0, 1, 1, -1, -1, -1, -1, -1])
relevance = [2^100, 1, 1, 1, 1, 1, 1, 1]

# Solve CVP
result = solve_cvp2(M, t, relevance, verbose=False)

# Extract coefficients
k = ZZ(round(result[1] / R[1,0]))
a0 = ZZ(round(-result[2] / R[2,1]))
a1 = ZZ(round(-result[3] / R[3,2]))
a2 = ZZ(round(-result[4] / R[4,3]))
b0 = ZZ(round(-result[5] / R[5,4]))
b1 = ZZ(round(-result[6] / R[6,5]))
b2 = ZZ(round(-result[7] / R[7,6]))

print(f"\n=== Decomposition Found ===")
print(f"k = {k} ({abs(k).nbits()} bits)")
if abs(k) > 1:
    print(f"  Note: k should be small, but got {k}")
print(f"\nCube root coefficients:")
print(f"  a0 = {a0} ({abs(a0).nbits()} bits)")
print(f"  a1 = {a1} ({abs(a1).nbits()} bits)")
print(f"  a2 = {a2} ({abs(a2).nbits()} bits)")
print(f"\nFrobenius coefficients:")
print(f"  b0 = {b0} ({abs(b0).nbits()} bits)")
print(f"  b1 = {b1} ({abs(b1).nbits()} bits)")
print(f"  b2 = {b2} ({abs(b2).nbits()} bits)")

# Size analysis
all_coeffs = [a0, a1, a2, b0, b1, b2]
max_bits = max(abs(c).nbits() for c in all_coeffs)
avg_bits = sum(abs(c).nbits() for c in all_coeffs) / 6

print(f"\n=== Size Analysis ===")
print(f"Original scalar: {s_bits_6d} bits")
print(f"Maximum coefficient: {max_bits} bits")
# print(f"Average coefficient: {avg_bits:.1f} bits")
print(f"Bit reduction: {s_bits_6d - max_bits} bits ({(s_bits_6d - max_bits)/s_bits_6d*100:.1f}%)")

# Verify the decomposition algebraically
print(f"\n=== Algebraic Verification ===")

# When k != 1, we need to account for it in the reconstruction
# The decomposition gives us: k*s = a0 + a1*λ_glv + ... 
# So s = (a0 + a1*λ_glv + ...) / k

if k != 0:
    # Compute the sum
    sum_without_k = Fr(a0 + a1*glv_lambda + a2*glv_lambda^2 + 
                       b0*lambda_frob + b1*lambda_frob*glv_lambda + 
                       b2*lambda_frob*glv_lambda^2)
    
    # Divide by k to get the original scalar
    reconstructed_6d = sum_without_k / k
    
    print(f"k*s = {sum_without_k}")
    print(f"s = (k*s) / k = {reconstructed_6d}")
else:
    print("Warning: k = 0, cannot reconstruct")
    reconstructed_6d = None

print(f"\nOriginal scalar: {s_random_6d}")
print(f"Reconstructed:   {reconstructed_6d}")
print(f"Match: {'✓ YES' if reconstructed_6d == s_random_6d else '✗ NO'}")

if true:
    print(f"\n=== G2 Point Verification (6D) ===")
    
    # Use a different test point for variety
    P = 3 * G2
    
    # Compute s*P directly
    print(f"Computing s*P directly...")
    Q_direct = ZZ(s_random_6d) * P
    
    print(f"Computing s*P using 6D decomposition...")
    
    # Compute the six basis points
    P0 = P                    # Identity
    P1 = psi(P)              # ψ(P)
    P2 = psi2(P)             # ψ²(P)
    P3 = frobenius_g2(P)     # φ(P)
    P4 = frobenius_g2(P1)    # φ(ψ(P))
    P5 = frobenius_g2(P2)    # φ(ψ²(P))
    
    print("Six basis points computed:")
    print("  P0 = P")
    print("  P1 = ψ(P)")
    print("  P2 = ψ²(P)")
    print("  P3 = φ(P)")
    print("  P4 = φ(ψ(P))")
    print("  P5 = φ(ψ²(P))")
    
    # Compute the linear combination
    print("\nComputing linear combination...")
    Q_decomp = E2(0)
    
    # Add each component with its coefficient
    if a0 != 0:
        Q_decomp = Q_decomp + abs(a0) * (P0 if a0 > 0 else -P0)
    if a1 != 0:
        Q_decomp = Q_decomp + abs(a1) * (P1 if a1 > 0 else -P1)
    if a2 != 0:
        Q_decomp = Q_decomp + abs(a2) * (P2 if a2 > 0 else -P2)
    if b0 != 0:
        Q_decomp = Q_decomp + abs(b0) * (P3 if b0 > 0 else -P3)
    if b1 != 0:
        Q_decomp = Q_decomp + abs(b1) * (P4 if b1 > 0 else -P4)
    if b2 != 0:
        Q_decomp = Q_decomp + abs(b2) * (P5 if b2 > 0 else -P5)
    
    # Verify they match
    print(f"\n=== Final Verification ===")
    print(f"Direct computation:     s*P")
    print(f"Decomposed computation: Σ aᵢ*Pᵢ")
    print(f"Points match: {'✓ YES' if Q_direct == Q_decomp else '✗ NO'}")
    
    if Q_direct == Q_decomp:
        print(f"\n🎉 Success! The full 6-dimensional endomorphism decomposition works correctly!")
        print(f"This confirms that:")
        print(f"  s·P = {a0}·P + {a1}·ψ(P) + {a2}·ψ²(P)")
        print(f"        + {b0}·φ(P) + {b1}·φ(ψ(P)) + {b2}·φ(ψ²(P))")
        print(f"\nCoefficient sizes:")
        for name, coeff in [('a0', a0), ('a1', a1), ('a2', a2), ('b0', b0), ('b1', b1), ('b2', b2)]:
            print(f"  {name}: {abs(coeff).nbits()} bits")
        print(f"\nThis reduces a {s_bits_6d}-bit scalar multiplication to")
        print(f"six {max_bits}-bit scalar multiplications!")
    else:
        print(f"\n⚠ Points don't match in 6D decomposition")
        print(f"Debugging information:")
        
        # Test individual components
        print(f"\nTesting cube root part only...")
        Q_cube = E2(0)
        if a0 != 0:
            Q_cube = Q_cube + abs(a0) * (P0 if a0 > 0 else -P0)
        if a1 != 0:
            Q_cube = Q_cube + abs(a1) * (P1 if a1 > 0 else -P1)
        if a2 != 0:
            Q_cube = Q_cube + abs(a2) * (P2 if a2 > 0 else -P2)
        
        s_cube = Fr(a0 + a1*glv_lambda + a2*glv_lambda^2)
        Q_cube_direct = ZZ(s_cube) * P
        print(f"Cube root part match: {'✓' if Q_cube_direct == Q_cube else '✗'}")
        
        print(f"\nTesting Frobenius part only...")
        Q_frob = E2(0)
        if b0 != 0:
            Q_frob = Q_frob + abs(b0) * (P3 if b0 > 0 else -P3)
        
        s_frob = Fr(b0*lambda_frob)
        Q_frob_direct = ZZ(s_frob) * P
        print(f"Frobenius part match: {'✓' if Q_frob_direct == Q_frob else '✗'}")
else:
    print(f"\n⚠ Algebraic verification failed")