use jolt_optimizations::{
    G2Projective, PrecomputedShamirTable, Windowed2CompactTable, Windowed2SignedTable,
    WindowedTable,
};
use std::mem::size_of;

fn main() {
    // Size of a single G2Projective point
    let g2_size = size_of::<G2Projective>();
    println!(
        "Size of G2Projective: {} bytes ({} bits)",
        g2_size,
        g2_size * 8
    );

    // Size of PrecomputedShamirTable
    let table_size = size_of::<PrecomputedShamirTable>();
    println!(
        "Size of PrecomputedShamirTable: {} bytes ({} bits)",
        table_size,
        table_size * 8
    );

    // Number of G2Projective points in the table
    let num_points = 256;
    println!("Number of G2Projective points in table: {}", num_points);

    // Calculate the multiple
    let multiple = table_size / g2_size;
    println!(
        "PrecomputedShamirTable is {}x the size of a single G2Projective",
        multiple
    );

    // Additional storage needed per point
    let additional_storage = table_size - g2_size;
    println!(
        "Additional storage per point: {} bytes ({} bits)",
        additional_storage,
        additional_storage * 8
    );
    println!(
        "Additional storage multiple: {}x",
        additional_storage / g2_size
    );

    // Size of kept table variants
    let windowed_table_size = size_of::<WindowedTable>();
    let windowed2_signed_table_size = size_of::<Windowed2SignedTable>();
    let windowed2_compact_table_size = size_of::<Windowed2CompactTable>();

    println!("\n=== Memory Usage Comparison (Kept Variants) ===");

    // Full table (256x) - baseline reference
    println!("Full PrecomputedShamirTable (baseline):");
    println!("  Size: {} bytes ({} bits)", table_size, table_size * 8);
    println!("  Multiple: {}x vs single point", table_size / g2_size);
    println!("  Entries: 256 (16 points × 16 signs)");

    // Windowed table (256x same as full) - performance champion
    println!("\nWindowed WindowedTable (performance champion):");
    println!(
        "  Size: {} bytes ({} bits)",
        windowed_table_size,
        windowed_table_size * 8
    );
    println!(
        "  Multiple: {}x vs single point",
        windowed_table_size / g2_size
    );
    println!("  Entries: 256 (4^4 = 256 windowed combinations)");
    println!("  Performance: ~30% faster than full table despite same memory");

    // 2-bit compact windowed table (64x) - balanced option
    println!("\n2-bit Compact Windowed2CompactTable (balanced option):");
    println!(
        "  Size: {} bytes ({} bits)",
        windowed2_compact_table_size,
        windowed2_compact_table_size * 8
    );
    println!(
        "  Multiple: {}x vs single point",
        windowed2_compact_table_size / g2_size
    );
    println!("  Entries: 64 (4^3 = 64 non-zero positive combinations)");
    println!(
        "  Memory reduction: {}x vs full ({}% smaller)",
        table_size / windowed2_compact_table_size,
        100 - (windowed2_compact_table_size * 100 / table_size)
    );

    // 2-bit signed windowed table (24x) - memory champion
    println!("\n2-bit Signed Windowed2SignedTable (memory champion):");
    println!(
        "  Size: {} bytes ({} bits)",
        windowed2_signed_table_size,
        windowed2_signed_table_size * 8
    );
    println!(
        "  Multiple: {}x vs single point",
        windowed2_signed_table_size / g2_size
    );
    println!("  Entries: 24 (4 bases × 6 variants each)");
    println!(
        "  Memory reduction: {}x vs full ({}% smaller)",
        table_size / windowed2_signed_table_size,
        100 - (windowed2_signed_table_size * 100 / table_size)
    );

    // Summary
    println!("\n=== Final Recommendations ===");
    println!("Single G2Projective:         {} bytes", g2_size);
    println!(
        "2-bit signed (24x):          {} bytes  <- Memory champion (90% savings)",
        windowed2_signed_table_size
    );
    println!(
        "2-bit compact (64x):         {} bytes  <- Balanced option (75% savings)",
        windowed2_compact_table_size
    );
    println!(
        "Full table (256x):           {} bytes  <- Baseline reference",
        table_size
    );
    println!(
        "Windowed table (256x):       {} bytes  <- Performance champion (30% faster)",
        windowed_table_size
    );

    println!("\n=== Usage Guidelines ===");
    println!("• Use windowed (256x) when performance is critical and memory isn't constrained");
    println!("• Use 2-bit compact (64x) for the best performance/memory balance");
    println!("• Use 2-bit signed (24x) when memory is very constrained");
    println!("• Use full table (256x) as baseline reference for comparisons");
}
