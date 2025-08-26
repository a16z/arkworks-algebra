use ark_bn254::{Fq, Fq12};
use ark_ff::{Field, One, UniformRand, Zero};
use ark_std::test_rng;

// Copy the test_helpers functions locally
use ark_ff::{BigInteger, PrimeField};

fn c18() -> Fq {
    Fq::from(18u64)
}
fn c82() -> Fq {
    Fq::from(82u64)
}
fn poly12_one() -> [Fq; 12] {
    let mut a = [Fq::zero(); 12];
    a[0] = Fq::one();
    a
}

fn mul_mod_g_12(a: &[Fq; 12], b: &[Fq; 12]) -> [Fq; 12] {
    let mut t = vec![Fq::zero(); 23];
    for i in 0..12 {
        if a[i].is_zero() {
            continue;
        }
        for j in 0..12 {
            if b[j].is_zero() {
                continue;
            }
            t[i + j] += a[i] * b[j];
        }
    }
    let k18 = c18();
    let k82 = c82();
    for k in (12..=22).rev() {
        let c = t[k];
        if c.is_zero() {
            continue;
        }
        t[k] = Fq::zero();
        t[k - 6] += k18 * c;
        t[k - 12] -= k82 * c;
    }
    let mut out = [Fq::zero(); 12];
    out.copy_from_slice(&t[0..12]);
    out
}

fn pow_mod_g_12(base: &[Fq; 12], exp: &Fq) -> [Fq; 12] {
    if exp.is_zero() {
        return poly12_one();
    }
    let mut res = poly12_one();
    let mut pwr = *base;
    let bits = Fq::MODULUS_BIT_SIZE as usize;
    let e = exp.into_bigint();
    for i in 0..bits {
        if e.get_bit(i) {
            res = mul_mod_g_12(&res, &pwr);
        }
        pwr = mul_mod_g_12(&pwr, &pwr);
    }
    res
}

fn poly_mul(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut result = vec![Fq::zero(); a.len() + b.len() - 1];
    for (i, a_i) in a.iter().enumerate() {
        for (j, b_j) in b.iter().enumerate() {
            result[i + j] += *a_i * *b_j;
        }
    }
    while result.len() > 1 && result.last() == Some(&Fq::zero()) {
        result.pop();
    }
    result
}

fn div_by_g_fold(mut r: Vec<Fq>) -> (Vec<Fq>, Vec<Fq>) {
    let mut q = vec![Fq::zero(); r.len().saturating_sub(12)];
    let k18 = c18();
    let k82 = c82();
    let mut k = r.len().saturating_sub(1);
    while k >= 12 {
        let c = r[k];
        if !c.is_zero() {
            q[k - 12] += c;
            r[k - 6] += k18 * c;
            r[k - 12] -= k82 * c;
            r[k] = Fq::zero();
        }
        if k == 12 {
            break;
        }
        k -= 1;
    }
    r.truncate(12);
    (q, r)
}

fn main() {
    let mut rng = test_rng();
    
    // Test a^3 = c
    let a = Fq12::rand(&mut rng);
    let c = a * a * a;
    
    println!("Testing a^3 = c");
    
    // Convert to polynomials
    use jolt_optimizations::fq12_to_poly12_coeffs;
    let a_poly = fq12_to_poly12_coeffs(&a);
    let c_poly = fq12_to_poly12_coeffs(&c);
    
    // Compute a^3 mod g
    let exp = Fq::from(3u64);
    let powered_a = pow_mod_g_12(&a_poly, &exp);
    
    println!("a_poly degree: {}", a_poly.iter().rposition(|x| !x.is_zero()).unwrap_or(0));
    println!("c_poly degree: {}", c_poly.iter().rposition(|x| !x.is_zero()).unwrap_or(0));
    println!("powered_a degree: {}", powered_a.iter().rposition(|x| !x.is_zero()).unwrap_or(0));
    
    // Check if powered_a equals c_poly
    let equal = powered_a.iter().zip(c_poly.iter()).all(|(x, y)| x == y);
    println!("powered_a == c_poly: {}", equal);
    
    // Compute residual = lhs - powered_a
    let mut residual = vec![Fq::zero(); 12];
    for i in 0..12 {
        residual[i] = c_poly[i] - powered_a[i];
    }
    
    // Check if residual is zero
    let residual_zero = residual.iter().all(|x| x.is_zero());
    println!("residual is zero: {}", residual_zero);
    
    if !residual_zero {
        println!("Non-zero residual coefficients:");
        for (i, coeff) in residual.iter().enumerate() {
            if !coeff.is_zero() {
                println!("  residual[{}] = {:?}", i, coeff);
            }
        }
        
        // Try dividing by g
        let (q, rem) = div_by_g_fold(residual.clone());
        println!("Quotient non-zero coeffs: {}", q.iter().filter(|x| !x.is_zero()).count());
        println!("Remainder zero: {}", rem.iter().all(|x| x.is_zero()));
        
        if !rem.iter().all(|x| x.is_zero()) {
            println!("Non-zero remainder coefficients:");
            for (i, coeff) in rem.iter().enumerate() {
                if !coeff.is_zero() {
                    println!("  rem[{}] = {:?}", i, coeff);
                }
            }
        }
    }
}