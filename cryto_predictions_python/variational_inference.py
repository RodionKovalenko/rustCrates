import numpy as np
from numpy.linalg import inv

def bayesian_linear_regression(X, y, prior_mean, prior_precision, num_iterations, tol=1e-6):
    n, d = X.shape
    posterior_mean = np.zeros((d, 1))
    posterior_precision = prior_precision
    
    identity = np.eye(d)
    
    for iter in range(num_iterations):
        # Store previous posterior_mean for convergence check
        previous_posterior_mean = posterior_mean.copy()
        
        # Update posterior parameters
        cov_inv = identity + posterior_precision * X.T.dot(X)
        cov = inv(cov_inv)
        posterior_mean = posterior_precision * cov.dot(X.T.dot(y) + posterior_mean * np.eye(d))
        
        # Update posterior precision
        posterior_precision = posterior_precision
        
        # Check convergence based on parameter change
        if np.max(np.abs(posterior_mean - previous_posterior_mean)) < tol:
            print(f"Converged after {iter+1} iterations.")
            break
    
    return posterior_mean, posterior_precision

def predict(x, posterior_mean, posterior_precision):
    mean_prediction = np.dot(x, posterior_mean)
    variance_prediction = 1.0 / posterior_precision + np.dot(x, np.dot(posterior_mean, x.T))
    return mean_prediction, variance_prediction

def main():
    # Generate some synthetic data
    np.random.seed(12)
    num_samples = 1000000
    num_features = 1
    true_slope = 5
    true_intercept = 1.0
    true_noise_std = 0.5
    
    X = np.random.randn(num_samples, num_features)
    noise = true_noise_std * np.random.randn(num_samples, 1)
    y_true = np.dot(X, np.array([[true_slope]])) + true_intercept + noise
    
    # Initialize prior parameters
    prior_mean = 0.0
    prior_precision = 1.0
    
    # Fit model using variational inference
    num_iterations = 1000
    posterior_mean, posterior_precision = bayesian_linear_regression(X, y_true, prior_mean, prior_precision, num_iterations)
    
    # Make predictions
    x_new = np.array([[1.0]])  # New data point to predict
    mean_prediction, variance_prediction = predict(x_new, posterior_mean, posterior_precision)
    
    print(f"Posterior mean: {mean_prediction}")
    print(f"Posterior variance: {variance_prediction}")

    # Cross-validation: Calculate mean squared error (MSE) on validation set
    X_valid = np.random.randn(20, num_features)
    noise_valid = true_noise_std * np.random.randn(20, 1)
    y_valid = np.dot(X_valid, np.array([[true_slope]])) + true_intercept + noise_valid

    y_pred_mean = np.dot(X_valid, posterior_mean)
    mse = np.mean((y_pred_mean - y_valid)**2)
    print(f"Validation MSE: {mse}")

if __name__ == "__main__":
    main()
