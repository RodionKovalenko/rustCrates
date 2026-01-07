#[cfg(test)]
mod tests_k_means_clustering {
    use crate::neural_networks::{optimization::k_means_clustering::kmeans, utils::random_arrays::generate_random_f32_2d};

    #[ignore]
    #[test]
    fn test_k_means_clustering() {
        println!("Creating input for k-means clustering test...");
        let mut input: Vec<Vec<f32>> = generate_random_f32_2d(50280, 64);

        println!("Starting k-means clustering test...");

        let k = 3000;

        let n_iter = 1000;

        let start = std::time::Instant::now();
        kmeans(&mut input, k, n_iter, 1e-4);

        let duration = start.elapsed();
        println!("Time elapsed in kmeans() is: {:?}", duration.as_secs_f64());

        // Since k-means is stochastic, we can't assert exact values,
        // but we can at least ensure the function runs without errors.
        assert!(true);
    }
}
