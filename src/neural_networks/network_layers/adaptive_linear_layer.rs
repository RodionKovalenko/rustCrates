use core::fmt::Debug;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;

use crate::neural_networks::utils::tokenizer::tokenize;
use crate::neural_networks::{
    network_components::{
        gradient_struct::Gradient,
        input::{load_data_xquad_de_as_dataset, load_data_xquad_en_as_dataset, load_data_xquad_ru_as_dataset, DataTrait, Dataset},
        layer_input_struct::LayerInput,
        layer_output_struct::LayerOutput,
    },
    network_layers::layer::LayerEnum,
    utils::{
        dtype::{r, C, ZERO},
        weights_initializer::initialize_weights_f32,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLinearLayer {
    // Cluster classifier (first level)
    pub cluster_weights: Vec<Vec<f32>>, // [num_clusters x hidden_size]
    pub cluster_bias: Vec<f32>,         // [num_clusters]

    // Word classifiers per cluster (second level)
    pub cluster_word_weights: Vec<Vec<Vec<f32>>>, // [cluster][word_in_cluster][hidden_size]
    pub cluster_word_bias: Vec<Vec<f32>>,         // [cluster][word_in_cluster]

    // Frequency‑based clusters
    pub frequency_clusters: Vec<Vec<usize>>, // cluster -> list of original token ids
    pub token_frequencies: Vec<u64>,         // for reference
    pub token_rank_by_id: Vec<usize>,        // rank of each token (0 = most frequent)
    pub token_cluster_by_id: Vec<usize>,     // cluster id for each token (0 = head)
    pub token_index_in_cluster: Vec<usize>,  // index inside its cluster for each token

    pub head_size: usize,
    pub tail_cluster_count: usize,

    // Hyper‑parameters
    pub learning_rate: f64,
    pub smoothing: f64,
    pub ema: f64,
    pub norm_layer: Option<LayerEnum>,
    pub global_norm: f64,
    pub max_norm: f64,

    // Forward/backward buffers
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub output_indices_batch: Vec<Vec<Vec<usize>>>,

    // Other fields from original (unused but kept for compatibility)
    pub weights: Vec<Vec<f32>>, // kept for serialization, not used
    pub previous_weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    pub gradient: Option<Gradient>,
    pub previous_gradient: Option<Gradient>,
    pub time_step: usize,
    pub batch_size: usize,
    pub selected_tail_cluster_batch: Vec<Vec<Option<usize>>>,
    pub target_batch_ids: Option<Vec<Vec<u32>>>,
}

impl AdaptiveLinearLayer {
    pub fn new(learning_rate: f64, hidden_size: usize, vocab_size: usize) -> Self {
        // Build frequency clusters
        let head_size = 2000;
        let tail_cluster_count = 18;
        let (token_frequencies, frequency_clusters, token_rank_by_id, token_cluster_by_id) = Self::build_frequency_clusters(vocab_size, head_size, tail_cluster_count);

        let num_clusters = frequency_clusters.len();

        // Build token -> index inside cluster mapping
        let mut token_index_in_cluster = vec![0; vocab_size];
        for (_cluster_id, tokens) in frequency_clusters.iter().enumerate() {
            for (pos, &tok) in tokens.iter().enumerate() {
                if tok < vocab_size {
                    token_index_in_cluster[tok] = pos;
                }
            }
        }

        // Initialize cluster classifier (first level)
        // 18x64
        let mut cluster_weights = vec![vec![0.0; hidden_size]; num_clusters];
        // 18
        let cluster_bias = vec![0.0; num_clusters];
        initialize_weights_f32(num_clusters, hidden_size, &mut cluster_weights);

        // Initialize word classifiers per cluster (second level)
        let mut cluster_word_weights = Vec::with_capacity(num_clusters);
        let mut cluster_word_bias = Vec::with_capacity(num_clusters);

        for tokens_in_cluster in &frequency_clusters {
            let cluster_size = tokens_in_cluster.len();
            let mut weights = vec![vec![0.0; hidden_size]; cluster_size];
            let bias = vec![0.0; cluster_size];
            initialize_weights_f32(cluster_size, hidden_size, &mut weights);
            cluster_word_weights.push(weights);
            cluster_word_bias.push(bias);
        }

        Self {
            cluster_weights,
            cluster_bias,
            cluster_word_weights,
            cluster_word_bias,
            frequency_clusters,
            token_frequencies,
            token_rank_by_id,
            token_cluster_by_id,
            token_index_in_cluster,
            head_size,
            tail_cluster_count: tail_cluster_count,
            learning_rate,
            smoothing: 0.99,
            ema: 0.0,
            norm_layer: None,
            global_norm: 0.0,
            max_norm: 0.0,
            input_batch: None,
            output_indices_batch: vec![],
            weights: vec![],
            previous_weights: vec![],
            bias: vec![],
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            batch_size: 0,
            selected_tail_cluster_batch: vec![],
            target_batch_ids: None,
        }
    }

    // ------------------------------------------------------------------------
    // Forward pass
    // ------------------------------------------------------------------------
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch = input.get_input_batch();
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();
        self.input_batch = Some(input_batch.clone());

        let target_tokens = input.get_target_batch_ids();
        let padding_mask_batch = input.get_padding_mask_batch();

        self.target_batch_ids = Some(target_tokens.clone());

        let batch_len = input_batch.len();
        let seq_len = input_batch[0].len();
        let num_clusters = self.frequency_clusters.len();

        let mut output_batch: Vec<Vec<Vec<C>>> = vec![vec![Vec::new(); seq_len]; batch_len];
        let mut output_indices_batch = vec![vec![Vec::new(); seq_len]; batch_len];

        self.output_indices_batch = output_indices_batch.clone();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);
        layer_output.set_output_indices(output_indices_batch);
        layer_output
    }

    // ------------------------------------------------------------------------
    // Backward pass
    // ------------------------------------------------------------------------
    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch missing");
        if input_batch.is_empty() {
            let mut grad = Gradient::new_default();
            grad.set_total_valid_tokens(previous_gradient.get_total_valid_tokens());
            self.gradient = Some(grad.clone());
            return grad;
        }

        let grad_combined_batch = previous_gradient.get_gradient_input_batch(); // dL/d(combined_logit)
        let total_valid_tokens = previous_gradient.get_total_valid_tokens();

        let batch_len = input_batch.len();
        let seq_len = input_batch[0].len();
        let hidden_size = input_batch[0][0].len();
        let num_clusters = self.frequency_clusters.len();
        let mut grad_input_batch = vec![vec![vec![C::new(ZERO, ZERO); hidden_size]; seq_len]; batch_len];

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(grad_input_batch);
        gradient.set_total_valid_tokens(total_valid_tokens);
        self.gradient = Some(gradient.clone());
        gradient
    }

    // ------------------------------------------------------------------------
    // Parameter update (AdamW with sparse updates)
    // ------------------------------------------------------------------------
    pub fn update_parameters(&mut self) {
        let total_valid_tokens = self.gradient.as_ref().map(|g| g.get_total_valid_tokens()).unwrap_or(1);
        let total_valid_tokens_real = r(total_valid_tokens.max(1) as f64);
        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        self.gradient = None;
    }

    // ------------------------------------------------------------------------
    // Frequency clustering (unchanged from original, but now also builds token_index_in_cluster)
    // ------------------------------------------------------------------------
    fn build_frequency_clusters(vocab_size: usize, head_size: usize, tail_cluster_count: usize) -> (Vec<u64>, Vec<Vec<usize>>, Vec<usize>, Vec<usize>) {
        let mut token_frequencies = vec![0_u64; vocab_size];

        let datasets = [
            load_data_xquad_de_as_dataset().unwrap(),
            load_data_xquad_en_as_dataset().unwrap(),
            load_data_xquad_ru_as_dataset().unwrap(),
        ];

        for dataset in &datasets {
            Self::accumulate_dataset_frequencies(dataset, &mut token_frequencies);
        }

        let mut ranked_token_ids: Vec<usize> = (0..vocab_size).collect();
        ranked_token_ids.sort_by_key(|&token_id| Reverse(token_frequencies[token_id]));

        let mut frequency_clusters = Vec::new();
        let head_cluster = ranked_token_ids.iter().take(head_size.min(vocab_size)).copied().collect::<Vec<_>>();
        frequency_clusters.push(head_cluster);

        let tail_tokens = &ranked_token_ids[head_size.min(vocab_size)..];
        let chunk_size = (tail_tokens.len() / tail_cluster_count.max(1)).max(1);
        for chunk in tail_tokens.chunks(chunk_size) {
            frequency_clusters.push(chunk.to_vec());
        }

        let mut token_rank_by_id = vec![0_usize; vocab_size];
        for (rank, token_id) in ranked_token_ids.into_iter().enumerate() {
            token_rank_by_id[token_id] = rank;
        }

        let mut token_cluster_by_id = vec![0_usize; vocab_size];
        for (cluster_id, cluster_tokens) in frequency_clusters.iter().enumerate() {
            for &token_id in cluster_tokens {
                token_cluster_by_id[token_id] = cluster_id;
            }
        }

        (token_frequencies, frequency_clusters, token_rank_by_id, token_cluster_by_id)
    }

    fn accumulate_dataset_frequencies(dataset: &Dataset<String, String>, token_frequencies: &mut Vec<u64>) {
        for text in dataset.get_input().iter().chain(dataset.get_target().iter()) {
            if let Ok((_, tokens)) = tokenize(text) {
                for tok in tokens {
                    if (tok as usize) < token_frequencies.len() {
                        token_frequencies[tok as usize] += 1;
                    }
                }
            }
        }
    }
}
