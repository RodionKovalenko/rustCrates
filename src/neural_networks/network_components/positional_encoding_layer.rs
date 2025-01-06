use num::Complex;
use serde::{Deserialize, Serialize};

use super::layer::BaseLayer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionalEncodingLayer {
    pub embedding_dim: usize, // Store the embedding dimension
}

impl PositionalEncodingLayer {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }
}

impl BaseLayer for PositionalEncodingLayer {
    /// Apply positional encoding to a batch of embeddings
    fn forward(&self, input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let mut output = Vec::with_capacity(input.len());

        for (position, token_embeddings) in input.iter().enumerate() {
            // Ensure all embeddings have the correct dimension
            assert_eq!(
                token_embeddings.len(),
                self.embedding_dim,
                "All token embeddings must have the same dimension as specified in the layer."
            );

            // Step 1: Add positional encodings to token embeddings (this could be a learned or fixed vector)
            let positional_encoding = generate_positional_encoding(position, self.embedding_dim);
            let token_with_pos_encoding = add_positional_encoding(token_embeddings, &positional_encoding);

            // Step 2: Apply rotary positional encoding
            let rotated_embeddings = apply_rotary_positional_encoding(&token_with_pos_encoding, position, self.embedding_dim);
            output.push(rotated_embeddings);
        }

        output
    }

    fn backward(&self, gradients: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        gradients.clone()
    }
}

/// Function to generate a positional encoding (for simplicity, we use a fixed formula here)
fn generate_positional_encoding(position: usize, embedding_dim: usize) -> Vec<f64> {
    let mut positional_encoding = vec![0.0; embedding_dim];
    for i in 0..embedding_dim {
        let angle = position as f64 / 10000.0_f64.powf(2.0 * i as f64 / embedding_dim as f64);
        positional_encoding[i] = angle.sin(); // Use sine as an example, but this can be adjusted
    }
    positional_encoding
}

/// Function to add positional encoding to token embeddings
fn add_positional_encoding(token_embeddings: &Vec<Complex<f64>>, positional_encoding: &Vec<f64>) -> Vec<Complex<f64>> {
    token_embeddings
        .iter()
        .zip(positional_encoding.iter())
        .map(|(embedding, pos_encoding)| {
            // Add positional encoding to the real part of the complex embedding (you can also apply to the imaginary part)
            Complex::new(embedding.re + pos_encoding, embedding.im)
        })
        .collect()
}

/// Function to apply rotary positional encoding to a single token's embedding
fn apply_rotary_positional_encoding(embedding: &Vec<Complex<f64>>, position: usize, embedding_dim: usize) -> Vec<Complex<f64>> {
    // Validate embedding length matches the embedding dimension
    assert_eq!(embedding.len(), embedding_dim, "Embedding length must match the specified embedding dimension.");

    // Ensure the embedding dimension is even
    assert_eq!(embedding_dim % 2, 0, "Embedding dimension must be even for rotary embeddings.");

    let mut rotated_embedding = embedding.clone(); // Initialize rotated embedding

    let half_dim = embedding_dim / 2;

    for i in 0..half_dim {
        // Compute the rotation angle theta for this pair of dimensions
        let theta = position as f64 / 10000.0_f64.powf(2.0 * i as f64 / embedding_dim as f64);
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // Apply the rotation to the pair of dimensions (even_idx, odd_idx)
        let even_idx = 2 * i;
        let odd_idx = even_idx + 1;

        let even = embedding[even_idx];
        let odd = embedding[odd_idx];

        // Rotate the real and imaginary parts for both even and odd indices
        rotated_embedding[even_idx] = Complex::new(even.re * cos_theta - odd.re * sin_theta, even.im * cos_theta - odd.im * sin_theta);
        rotated_embedding[odd_idx] = Complex::new(even.re * sin_theta + odd.re * cos_theta, even.im * sin_theta + odd.im * cos_theta);
    }

    rotated_embedding
}
