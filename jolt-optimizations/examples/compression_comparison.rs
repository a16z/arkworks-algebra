use ark_bn254::{Bn254, Fq12, G1Projective, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup};
use ark_ff::UniformRand;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use ark_std::test_rng;

fn main() {
    let mut rng = test_rng();

    println!("BN254 Compression Comparison");
    println!("{}", "=".repeat(50));

    println!("\nG1 Element:");
    let g1_element = G1Projective::rand(&mut rng);
    compare_compression(g1_element, "G1");

    println!("\nG2 Element:");
    let g2_element = G2Projective::rand(&mut rng);
    compare_compression(g2_element, "G2");

    println!("\nGT Element (Fq12):");
    let g1_affine = g1_element.into_affine();
    let g2_affine = g2_element.into_affine();
    let gt_element = Bn254::pairing(g1_affine, g2_affine);
    compare_compression_fq12(gt_element.0, "GT");
}

fn compare_compression<T: CanonicalSerialize + CanonicalDeserialize>(element: T, _name: &str) {
    // Uncompressed serialization
    let mut uncompressed_bytes = Vec::new();
    element
        .serialize_with_mode(&mut uncompressed_bytes, Compress::No)
        .expect("Failed to serialize uncompressed");
    let uncompressed_size = uncompressed_bytes.len();

    // Compressed serialization
    let mut compressed_bytes = Vec::new();
    element
        .serialize_with_mode(&mut compressed_bytes, Compress::Yes)
        .expect("Failed to serialize compressed");
    let compressed_size = compressed_bytes.len();

    let compression_ratio = (compressed_size as f64 / uncompressed_size as f64) * 100.0;
    let size_reduction = 100.0 - compression_ratio;

    println!("  Uncompressed size: {} bytes", uncompressed_size);
    println!("  Compressed size:   {} bytes", compressed_size);
    println!("  Compression ratio: {:.2}%", compression_ratio);
    println!("  Size reduction:    {:.2}%", size_reduction);

    // Verify we can deserialize
    let _deserialized: T =
        T::deserialize_with_mode(&compressed_bytes[..], Compress::Yes, Validate::Yes)
            .expect("Failed to deserialize compressed data");
}

fn compare_compression_fq12(element: Fq12, _name: &str) {
    // Uncompressed serialization
    let mut uncompressed_bytes = Vec::new();
    element
        .serialize_with_mode(&mut uncompressed_bytes, Compress::No)
        .expect("Failed to serialize uncompressed");
    let uncompressed_size = uncompressed_bytes.len();

    // Compressed serialization
    let mut compressed_bytes = Vec::new();
    element
        .serialize_with_mode(&mut compressed_bytes, Compress::Yes)
        .expect("Failed to serialize compressed");
    let compressed_size = compressed_bytes.len();

    // Calculate compression ratio
    let compression_ratio = (compressed_size as f64 / uncompressed_size as f64) * 100.0;
    let size_reduction = 100.0 - compression_ratio;

    println!("  Uncompressed size: {} bytes", uncompressed_size);
    println!("  Compressed size:   {} bytes", compressed_size);
    println!("  Compression ratio: {:.2}%", compression_ratio);
    println!("  Size reduction:    {:.2}%", size_reduction);

    // Verify we can deserialize
    let _deserialized: Fq12 =
        Fq12::deserialize_with_mode(&compressed_bytes[..], Compress::Yes, Validate::Yes)
            .expect("Failed to deserialize compressed data");
}
