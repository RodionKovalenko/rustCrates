use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::neural_networks::utils::{dtype::C, shared_f32_matrix::SharedF32Matrix};

/// Shared (tied) embedding table + a cross-layer gradient accumulator.
///
/// Intended usage: `SparseLinearLayer` owns the optimizer update, while `EmbeddingLayer`
/// only accumulates per-token gradients into `grad_by_token`.
#[derive(Clone, Debug)]
pub struct TiedSparseEmbeddings {
    pub weights: SharedF32Matrix,
    pub grad_by_token: Arc<RwLock<HashMap<usize, Vec<C>>>>,
}

impl TiedSparseEmbeddings {
    pub fn new(weights: SharedF32Matrix) -> Self {
        Self {
            weights,
            grad_by_token: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}
