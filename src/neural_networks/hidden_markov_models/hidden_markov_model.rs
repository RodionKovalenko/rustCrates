use num_complex::Complex;

pub struct HMM {
    // Initial state probabilities (π)
    initial: Vec<Complex<f64>>,
    // State transition matrix (A)
    transition: Vec<Vec<Complex<f64>>>,
    // Observation emission matrix (B)
    emission: Vec<Vec<Complex<f64>>>,
}

impl HMM {
    pub fn new(
        initial: Vec<Complex<f64>>,
        transition: Vec<Vec<Complex<f64>>>,
        emission: Vec<Vec<Complex<f64>>>,
    ) -> Self {
        // Validate dimensions
        let states = initial.len();
        assert_eq!(transition.len(), states);
        assert!(transition.iter().all(|row| row.len() == states));
        assert_eq!(emission.len(), states);
        
        HMM {
            initial,
            transition,
            emission,
        }
    }

    /// Forward algorithm (computes probability of observations)
    pub fn forward(&self, observations: &[usize]) -> Complex<f64> {
        let states = self.initial.len();
        let mut alpha = vec![Complex::new(0.0, 0.0); states];
        
        // Initialize with first observation
        for s in 0..states {
            alpha[s] = self.initial[s] * self.emission[s][observations[0]];
        }
        
        // Iterate through remaining observations
        for &obs in &observations[1..] {
            let mut new_alpha = vec![Complex::new(0.0, 0.0); states];
            
            for j in 0..states {
                let sum: Complex<f64> = (0..states)
                    .map(|i| alpha[i] * self.transition[i][j])
                    .sum();
                
                new_alpha[j] = sum * self.emission[j][obs];
            }
            
            alpha = new_alpha;
        }
        
        alpha.iter().sum()
    }

    /// Viterbi algorithm (finds most likely state sequence)
    pub fn viterbi(&self, observations: &[usize]) -> Vec<usize> {
        let states = self.initial.len();
        let t_len = observations.len();
        let mut v = vec![vec![Complex::new(0.0, 0.0); states]; t_len];
        let mut backpointers = vec![vec![0; states]; t_len];
        
        // Initialize
        for s in 0..states {
            v[0][s] = self.initial[s] * self.emission[s][observations[0]];
        }
        
        // Recursion
        for t in 1..t_len {
            for s in 0..states {
                let max = (0..states)
                    .map(|prev_s| v[t-1][prev_s] * self.transition[prev_s][s])
                    .max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap())
                    .unwrap();
                
                v[t][s] = max * self.emission[s][observations[t]];
                backpointers[t][s] = (0..states)
                    .max_by(|&a, &b| {
                        (v[t-1][a] * self.transition[a][s])
                            .norm()
                            .partial_cmp(&(v[t-1][b] * self.transition[b][s]).norm())
                            .unwrap()
                    })
                    .unwrap();
            }
        }
        
        // Backtracking
        let mut path = vec![0; t_len];
        path[t_len-1] = (0..states)
            .max_by(|&a, &b| v[t_len-1][a].norm().partial_cmp(&v[t_len-1][b].norm()).unwrap())
            .unwrap();
        
        for t in (0..t_len-1).rev() {
            path[t] = backpointers[t+1][path[t+1]];
        }
        
        path
    }
}

// Example usage
pub fn test_hmm() {
    use num_complex::Complex;

    // Create a simple HMM with complex numbers
    let initial = vec![
        Complex::new(0.6, 0.0),
        Complex::new(0.4, 0.0)
    ];
    
    let transition = vec![
        vec![Complex::new(0.7, 0.0), Complex::new(0.3, 0.0)],
        vec![Complex::new(0.4, 0.0), Complex::new(0.6, 0.0)]
    ];
    
    let emission = vec![
        vec![Complex::new(0.1, 0.0), Complex::new(0.4, 0.0), Complex::new(0.5, 0.0)],
        vec![Complex::new(0.6, 0.0), Complex::new(0.3, 0.0), Complex::new(0.1, 0.0)]
    ];
    
    let hmm = HMM::new(initial, transition, emission);
    
    // Example observation sequence
    let observations = vec![0, 1, 2];
    
    // Compute probability
    let prob = hmm.forward(&observations);
    println!("Probability of observations: {}", prob);
    
    // Find most likely path
    let path = hmm.viterbi(&observations);
    println!("Most likely state sequence: {:?}", path);
}