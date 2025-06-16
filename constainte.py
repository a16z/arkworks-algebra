# BN128 elliptic curve
p  = 21888242871839275222246405745257275088696311157297823662689037894645226208583
r  = 21888242871839275222246405745257275088548364400416034343698204186575808495617
h  = 1
Fp = GF(p)
Fr = GF(r)
A  = Fp(0)
B  = Fp(3)
E  = EllipticCurve(Fp,[Name,B])
gx = Fp(1)
gy = Fp(2)
gen = E(gx,gy)  # subgroup generator
print("scalar field check: ", gen.additive_order() == r )
print("cofactor check:     ", E.cardinality() == r*h )

# extension field
R.<x>   = Fp[]
Fp2.<u> = Fp.extension(x^2+1)

# twisted curve
B_twist = Fp2(19485874751759354771024239261021720505790618469301721065564631296452457478373 + 266929791119991161246907387137283842545076965332900288569378510910307636690*u )
E2 = EllipticCurve(Fp2,[0,B_twist])
size_E2     = E2.cardinality();
cofactor_E2 = size_E2 / r;

gen2_xi = Fp( 0x1adcd0ed10df9cb87040f46655e3808f98aa68a570acf5b0bde23fab1f149701 )
gen2_xu = Fp( 0x09e847e9f05a6082c3cd2a1d0a3a82e6fbfbe620f7f31269fa15d21c1c13b23b )
gen2_yi = Fp( 0x056c01168a5319461f7ca7aa19d4fcfd1c7cdf52dbfc4cbee6f915250b7f6fc8 )
gen2_yu = Fp( 0x0efe500a2d02dd77f5f401329f30895df553b878fc3c0dadaaa86456a623235c )

gen2_x = gen2_xi + u * gen2_xu
gen2_y = gen2_yi + u * gen2_yu

gen2 = E2(gen2_x, gen2_y)

print("g2^r: ", gen2*r )

expo = 0x7b17fcc286b01af79176aa7da3a8615020eacda89a90e4ff5d0a085483f0448

print("g2^expo: ")
print(gen2*expo)