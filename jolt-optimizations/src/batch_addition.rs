//! Batch affine point addition for G1

use ark_bn254::G1Affine;
use ark_std::Zero;

/// Performs batch addition of G1 affine points.
///
/// # Arguments
/// * `bases` - Slice of G1 affine points to select from
/// * `indices` - Slice of indices specifying which points to sum
///
/// # Returns
/// The sum of all selected points as a single G1Affine point
pub fn batch_g1_additions(bases: &[G1Affine], indices: &[usize]) -> G1Affine {
    if indices.is_empty() {
        return G1Affine::identity();
    }

    if indices.len() == 1 {
        return bases[indices[0]];
    }

    let mut points: Vec<G1Affine> = Vec::with_capacity(indices.len());
    points.extend(indices.iter().map(|&i| bases[i]));

    while points.len() > 1 {
        let current_len = points.len();
        let pairs_count = current_len / 2;
        let has_odd = current_len % 2 == 1;

        let denominators: Vec<_> = (0..pairs_count)
            .map(|i| {
                let p1 = points[i * 2];
                let p2 = points[i * 2 + 1];
                p2.x - p1.x
            })
            .collect();

        let mut inverses = denominators;
        ark_ff::fields::batch_inversion(&mut inverses);

        let mut new_points: Vec<G1Affine> = (0..pairs_count)
            .zip(inverses.iter())
            .map(|(i, inv)| {
                let p1 = points[i * 2];
                let p2 = points[i * 2 + 1];
                let lambda = (p2.y - p1.y) * inv;
                let x3 = lambda * lambda - p1.x - p2.x;
                let y3 = lambda * (p1.x - x3) - p1.y;
                G1Affine::new_unchecked(x3, y3)
            })
            .collect();

        if has_odd {
            new_points.push(points[current_len - 1]);
        }

        points = new_points;
    }

    points[0]
}

/// Performs multiple batch additions of G1 affine points.
/// Uses projective running sums with a single batch normalization at the end.
pub fn batch_g1_additions_multi(bases: &[G1Affine], indices_sets: &[Vec<usize>]) -> Vec<G1Affine> {
    use ark_bn254::G1Projective;
    use ark_ec::CurveGroup;

    if indices_sets.is_empty() {
        return vec![];
    }

    let proj_sums: Vec<G1Projective> = indices_sets
        .iter()
        .map(|indices| {
            if indices.is_empty() {
                return G1Projective::zero();
            }
            let mut sum = G1Projective::from(bases[indices[0]]);
            for &idx in &indices[1..] {
                sum += bases[idx];
            }
            sum
        })
        .collect();

    G1Projective::normalize_batch(&proj_sums)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::AffineRepr;
    use ark_std::rand::RngCore;
    use ark_std::UniformRand;

    #[test]
    fn test_batch_addition_correctness() {
        let mut rng = ark_std::test_rng();

        let bases: Vec<G1Affine> = (0..10).map(|_| G1Affine::rand(&mut rng)).collect();

        let indices = vec![2, 3, 4, 5, 6, 7];

        let batch_result = batch_g1_additions(&bases, &indices);

        let mut expected = G1Affine::identity();
        for &idx in &indices {
            expected = (expected + bases[idx]).into();
        }

        assert_eq!(batch_result, expected);
    }

    #[test]
    fn test_empty_indices() {
        let bases: Vec<G1Affine> = vec![G1Affine::generator(); 5];
        let result = batch_g1_additions(&bases, &[]);
        assert_eq!(result, G1Affine::identity());
    }

    #[test]
    fn test_single_index() {
        let mut rng = ark_std::test_rng();
        let bases: Vec<G1Affine> = (0..5).map(|_| G1Affine::rand(&mut rng)).collect();

        let result = batch_g1_additions(&bases, &[2]);
        assert_eq!(result, bases[2]);
    }

    #[test]
    fn test_batch_additions_multi() {
        let mut rng = ark_std::test_rng();

        let base_size = 10000;
        let num_batches = 50;

        let bases: Vec<G1Affine> = (0..base_size).map(|_| G1Affine::rand(&mut rng)).collect();

        let indices_sets: Vec<Vec<usize>> = (0..num_batches)
            .map(|_| {
                let size = (rng.next_u64() as usize) % 100 + 1;
                (0..size)
                    .map(|_| (rng.next_u64() as usize) % base_size)
                    .collect()
            })
            .collect();

        let batch_results = batch_g1_additions_multi(&bases, &indices_sets);

        for (i, (result, indices)) in batch_results.iter().zip(indices_sets.iter()).enumerate() {
            let single_result = batch_g1_additions(&bases, indices);
            assert_eq!(
                *result, single_result,
                "Multi vs single mismatch at batch {}",
                i
            );
        }
    }
}
