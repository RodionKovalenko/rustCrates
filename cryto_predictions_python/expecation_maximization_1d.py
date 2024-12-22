import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the Gaussian probability density function (PDF)
def gaussian_pdf(x, mean, variance):
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-(x - mean)**2 / (2 * variance))

# EM Algorithm for 1D Gaussian Mixture Model
def em_algorithm_1d(data, num_components=2, max_iter=100, tol=1e-6):
    n = len(data)
    
    # Initialize parameters: means, variances, and weights for the components
    np.random.seed(42)
    means = np.random.choice(data, num_components)
    variances = np.random.random(num_components)

    print("means: ", means)
    print("variances: ", variances)
    weights = np.ones(num_components) / num_components
    
    log_likelihoods = []
    
    for iteration in range(max_iter):
        # E-step: Calculate responsibilities
        responsibilities = np.zeros((n, num_components))
        
        for i in range(num_components):
            responsibilities[:, i] = weights[i] * gaussian_pdf(data, means[i], variances[i])
        
        # Normalize responsibilities (to sum to 1 for each data point)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        # M-step: Update parameters
        new_means = np.sum(responsibilities * data[:, None], axis=0) / responsibilities.sum(axis=0)
        new_variances = np.sum(responsibilities * (data[:, None] - new_means)**2, axis=0) / responsibilities.sum(axis=0)
        new_weights = responsibilities.sum(axis=0) / n
        
        # Calculate the log-likelihood
        gaussian_pdfs = np.array([gaussian_pdf(data, new_means[i], new_variances[i]) for i in range(num_components)])
        log_likelihood = np.sum(np.log(np.sum(responsibilities * gaussian_pdfs.T, axis=1)))
        log_likelihoods.append(log_likelihood)
        
        # Check for convergence (change in log-likelihood)
        if iteration > 0 and np.abs(log_likelihood - log_likelihoods[-2]) < tol:
            print(f"Converged at iteration {iteration}.")
            break
        
        # Update parameters
        means = new_means
        variances = new_variances
        weights = new_weights
        
    return means, variances, weights, log_likelihoods

# Sample 1D data (for example, generated from two Gaussian distributions)
data = np.concatenate([np.random.normal(5, 1, 200), np.random.normal(10, 2, 300)])

# Run the EM algorithm
means, variances, weights, log_likelihoods = em_algorithm_1d(data, num_components=2)

# Print the estimated parameters
print("Estimated Means:", means)
print("Estimated Variances:", variances)
print("Estimated Weights:", weights)

# Plot the data and the estimated Gaussian components
x = np.linspace(min(data), max(data), 1000)
pdf = np.sum([weights[i] * gaussian_pdf(x, means[i], variances[i]) for i in range(2)], axis=0)

plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data')
plt.plot(x, pdf, label='Fitted GMM', color='r')
plt.legend()
plt.show()

# Plot log-likelihood curve
plt.plot(log_likelihoods)
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood Convergence')
plt.show()
