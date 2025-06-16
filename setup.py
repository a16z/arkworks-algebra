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
g2_x = Fp(10857046999023057135944570762232829481370756359578518086990519993285655852781) + \
       Fp(11559732032986387107991004021392285783925812861821192530917403151452391805634) * u
g2_y = Fp(8495653923123431417604973247489272438418190587263600148770280649306958101930) + \
       Fp(4082367875863433681332203403145435568316851327593401208105741076214120093531) * u

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
PSI_X = Fp(21575463638280843010398324269430826099269044274347216827212613867836435027261) + \
        Fp(10307601595873709700152284273816112264069230130616436755625194854815875713954) * u

# PSI_Y = (u+9)^((p-1)/2)
PSI_Y = Fp(2821565182194536844548159561693502659359617185244120367078079554186484126554) + \
        Fp(3505843767911556378687030309984248845540243509899259641013678093033130930403) * u

print(f"\nFrobenius coefficients:")
print(f"PSI_X = xi^((p-1)/3)")
print(f"PSI_Y = xi^((p-1)/2)")

# Verify these are correct
assert PSI_X == xi^((p-1)//3), "PSI_X mismatch"
assert PSI_Y == xi^((p-1)//2), "PSI_Y mismatch"
print("✓ Frobenius coefficients verified")

# GLV endomorphism coefficient from Rust code
GLV_ENDO_COEFF = Fp(21888242871839275220042445260109153167277707414472061641714758635765020556616) + \
                 Fp(0) * u

# This is -1 in Fp, which gives us the cube root of unity for the endomorphism
omega_glv = GLV_ENDO_COEFF
assert omega_glv^3 == 1 and omega_glv != 1, "GLV coefficient should be a cube root of unity"

print(f"\nGLV endomorphism coefficient: {omega_glv}")
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
psi_eigentest = ZZ(zeta3) * test_point

print(f"Testing ψ endomorphism...")
# The GLV lambda from the Rust code
glv_lambda = Fr(4407920970296243842393367215006156084916469457145843978461)
psi_eigen_test = ZZ(glv_lambda) * test_point
print(f"ψ acts with eigenvalue λ_glv = {glv_lambda}")
assert psi_test == psi_eigen_test, "ψ endomorphism failed"
print("✓ ψ endomorphism verified")

# Test Frobenius
print(f"\nTesting Frobenius endomorphism...")
frob_test = frobenius_g2(test_point)
frob_eigen_test = ZZ(lambda_frob) * test_point
assert frob_test == frob_eigen_test, "Frobenius endomorphism failed"
print("✓ Frobenius endomorphism verified")

# Now let's test the decomposition with the corrected endomorphisms
print(f"\n=== Scalar Decomposition ===")

# For the full 6-dimensional decomposition, we need to use the correct eigenvalues
# The basis is: 1, ψ, ψ², φ, φψ, φψ²
# With eigenvalues: 1, λ_glv, λ_glv², p, p*λ_glv, p*λ_glv²

# Generate random scalar
s_random = Fr.random_element()
s_bits = ZZ(s_random).nbits()
print(f"Random scalar: {s_bits} bits")

# We'll use a simpler 2-dimensional GLV decomposition for demonstration
# This uses just the cube root endomorphism: s = a0 + a1*λ_glv

# Build the 2-dimensional GLV lattice
L_glv = Matrix([
    [1, 0],
    [0, 1],
    [ZZ(glv_lambda), 0],
    [0, r]
])

# Find short vector
v = L_glv.LLL()[0]
a1_base = ZZ(v[0])
a0_base = ZZ(v[1])

# Decompose s
k1 = (ZZ(s_random) * a1_base) // r
k0 = -((ZZ(s_random) * a0_base) // r)

a0 = ZZ(s_random) - k0 * ZZ(glv_lambda)
a1 = -k0

print(f"\n=== 2-GLV Decomposition ===")
print(f"s = {a0} + {a1}*λ_glv")
print(f"a0: {abs(a0).nbits()} bits")
print(f"a1: {abs(a1).nbits()} bits")

# Verify algebraically
reconstructed = Fr(a0 + a1 * glv_lambda)
print(f"Algebraic verification: {'✓' if reconstructed == s_random else '✗'}")

# Verify on curve
print(f"\n=== Curve Verification ===")
Q_direct = ZZ(s_random) * G2
Q_decomp = abs(a0) * (G2 if a0 > 0 else -G2)
if a1 != 0:
    Q_decomp += abs(a1) * (psi(G2) if a1 > 0 else -psi(G2))

print(f"Direct computation vs decomposition: {'✓' if Q_direct == Q_decomp else '✗'}")

if Q_direct == Q_decomp:
    print(f"\n🎉 Success! The endomorphism decomposition works correctly!")
    print(f"This confirms that: s·P = {a0}·P + {a1}·ψ(P)")
    print(f"Bit savings: {s_bits - max(abs(a0).nbits(), abs(a1).nbits())} bits")
    
# Additional: demonstrate the Frobenius endomorphism works
print(f"\n=== Frobenius Endomorphism Demo ===")
# For any point P, we have φ(P) = p*P in the r-torsion
test_p = 123 * G2
frob_p = frobenius_g2(test_p)
p_times_p = ZZ(lambda_frob) * test_p
print(f"φ(P) = p·P (mod r): {'✓' if frob_p == p_times_p else '✗'}")

print(f"\nNote: For a full 6-dimensional decomposition, you would extend")
print(f"the lattice to include both ψ and φ endomorphisms, giving")
print(f"coefficients that are ~85-90 bits instead of ~254 bits.")