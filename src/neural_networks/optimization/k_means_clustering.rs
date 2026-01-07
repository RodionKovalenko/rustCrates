use std::fmt::Debug;

use num_traits::{Float, FromPrimitive};
use rand::seq::IndexedRandom;

// ------------------------------------------------------------
// L2 normalization helper
// ------------------------------------------------------------
fn l2_normalize<T: Float>(v: &mut [T]) {
    let mut norm = T::zero();
    for &x in v.iter() {
        norm = norm + x * x;
    }

    let epsilon = T::from(1e-12).unwrap();
    norm = norm.sqrt().max(epsilon);

    for x in v.iter_mut() {
        *x = *x / norm;
    }
}

// ------------------------------------------------------------
// compute squared L2 distance between two vectors
// ------------------------------------------------------------
fn squared_l2_distance<T: Float>(a: &[T], b: &[T]) -> T {
    let mut sum = T::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = *x - *y;
        sum = sum + diff * diff;
    }
    sum
}

// ------------------------------------------------------------
// k-means with L2 normalization and early stopping
// ------------------------------------------------------------
pub fn kmeans<T: Float + FromPrimitive + Debug>(
    data: &mut Vec<Vec<T>>, // N x D
    k: usize,
    n_iter: usize,
    tol: T, // early stopping threshold
) -> (Vec<Vec<T>>, Vec<usize>, Vec<Vec<usize>>) {
    let n = data.len();
    let d = data[0].len();

    assert!(n >= k);
    println!("Running K-means with N={}, D={}, K={}", n, d, k);

    // normalize data ONCE
    for v in data.iter_mut() {
        l2_normalize(v);
    }

    // init centroids
    let mut rng = rand::rng();
    let mut centroids: Vec<Vec<T>> = data.as_slice().choose_multiple(&mut rng, k).map(|v| v.clone()).collect();

    let mut assignments = vec![0usize; n];
    let mut old_centroids = centroids.clone();

    let target_size = n / k;

    for iter in 0..n_iter {
        let mut sums = vec![vec![T::zero(); d]; k];
        let mut counts = vec![0usize; k];

        // annealed penalty strength
        let alpha = {
            let t = T::from(iter).unwrap() / T::from(n_iter).unwrap();
            T::from(0.1).unwrap() + T::from(0.9).unwrap() * t
        };

        // ------------------------
        // assignment step (with penalty)
        // ------------------------
        for (i, v) in data.iter().enumerate() {
            let mut best_k = 0;
            let mut best_score = T::max_value();

            for (c_idx, c) in centroids.iter().enumerate() {
                let dist = squared_l2_distance(v, c);

                // size penalty
                let size_ratio = T::from(counts[c_idx]).unwrap() / T::from(target_size.max(1)).unwrap();

                let score = dist * (T::one() + alpha * size_ratio);

                if score < best_score {
                    best_score = score;
                    best_k = c_idx;
                }
            }

            // Assignment of token index i to cluster best_k
            assignments[i] = best_k;
            counts[best_k] += 1;

            for j in 0..d {
                sums[best_k][j] = sums[best_k][j] + v[j];
            }
        }

        // ------------------------
        // update step
        // ------------------------
        for c_idx in 0..k {
            if counts[c_idx] == 0 {
                continue;
            }

            let inv = T::one() / T::from(counts[c_idx]).unwrap();
            for j in 0..d {
                centroids[c_idx][j] = sums[c_idx][j] * inv;
            }

            l2_normalize(&mut centroids[c_idx]);
        }

        // ------------------------
        // check convergence
        // ------------------------
        let mut max_shift = T::zero();
        for (new, old) in centroids.iter().zip(old_centroids.iter()) {
            let shift = squared_l2_distance(new, old).sqrt();
            if shift > max_shift {
                max_shift = shift;
            }
        }

        if max_shift < tol {
            println!("K-means converged with max centroid shift {:?} at iteration {}", max_shift, iter);
            break;
        }

        old_centroids = centroids.clone();
    }

    // Return also the tokens assigned to each cluster
    let mut cluster_to_tokens = vec![Vec::new(); k];
    for (token_id, &c) in assignments.iter().enumerate() {
        cluster_to_tokens[c].push(token_id);
    }

    println!("K-means completed.");

    (centroids, assignments, cluster_to_tokens)
}

// ------------------------------------------------------------
// Query function: returns candidate token IDs
// ------------------------------------------------------------
pub fn query_candidates<T: Float>(
    query: &mut [T],                  // query vector
    centroids: &[Vec<T>],             // cluster centroids
    cluster_to_tokens: &[Vec<usize>], // precomputed cluster → token IDs
    top_m_centroids: usize,           // how many clusters to select
) -> Vec<usize> {
    let k = centroids.len();
    let d = query.len();

    // normalize query vector
    let mut norm = T::zero();
    for &x in query.iter() {
        norm = norm + x * x;
    }
    let epsilon = T::from(1e-12).unwrap();
    norm = norm.sqrt().max(epsilon);
    for x in query.iter_mut() {
        *x = *x / norm;
    }

    // score each centroid (using cosine similarity = dot product for L2-normalized vectors)
    let mut centroid_scores: Vec<(usize, T)> = Vec::with_capacity(k);
    for (c_idx, c) in centroids.iter().enumerate() {
        let mut score = T::zero();
        for j in 0..d {
            score = score + query[j] * c[j];
        }
        centroid_scores.push((c_idx, score));
    }

    // pick top_m_centroids
    centroid_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_clusters: Vec<usize> = centroid_scores.iter().take(top_m_centroids).map(|&(c_idx, _)| c_idx).collect();

    // expand tokens from top clusters
    let mut candidates = Vec::new();
    for &c in top_clusters.iter() {
        candidates.extend(&cluster_to_tokens[c]);
    }

    candidates
}
