use std::mem::size_of;
use jolt_optimizations::{G2Projective, PrecomputedShamirTable};

fn main() {
    // Size of a single G2Projective point
    let g2_size = size_of::<G2Projective>();
    println!("Size of G2Projective: {} bytes ({} bits)", g2_size, g2_size * 8);
    
    // Size of PrecomputedShamirTable
    let table_size = size_of::<PrecomputedShamirTable>();
    println!("Size of PrecomputedShamirTable: {} bytes ({} bits)", table_size, table_size * 8);
    
    // Number of G2Projective points in the table
    let num_points = 256;
    println!("Number of G2Projective points in table: {}", num_points);
    
    // Calculate the multiple
    let multiple = table_size / g2_size;
    println!("PrecomputedShamirTable is {}x the size of a single G2Projective", multiple);
    
    // Additional storage needed per point
    let additional_storage = table_size - g2_size;
    println!("Additional storage per point: {} bytes ({} bits)", additional_storage, additional_storage * 8);
    println!("Additional storage multiple: {}x", additional_storage / g2_size);
    
    // Alternative calculation: 256 points vs 1 point
    println!("\nBreakdown:");
    println!("- Original point: 1 × {} bytes = {} bytes", g2_size, g2_size);
    println!("- Precomputed table: 256 × {} bytes = {} bytes", g2_size, 256 * g2_size);
    println!("- Storage multiple: 256x");
    println!("- Additional storage: {} bytes ({} bits)", 255 * g2_size, 255 * g2_size * 8);
}