import numpy as np
from scipy.stats import multivariate_normal

# Generate synthetic 2D data
data = np.array([[0.67757034, 3.2006002],
            [2.31985125, 3.08571429],
            [0.24864422, 2.01607895],
            [2.13529672, 3.67785677]])

# Number of clusters
k = 2

# Initialize parameters
n, d = data.shape
pi = np.ones(k) / k  # Equal mixing coefficients
mu = np.array([[0.5, 2.5], [2.0, 3.5]])  # Initial means
sigma = np.array([
    [[1.0, 0.5], [0.5, 1.0]],
    [[1.5, 0.5], [0.5, 1.5]],
    ])  # Initial covariances

print("pi", pi);
print("mu", mu);
print("sigma", sigma)
print("end")

# EM Algorithm
def em_gmm(data, k, max_iter=100, tol=1e-6):
    n, d = data.shape
    pi = np.ones(k) / k
    mu = np.random.rand(k, d)
    sigma = np.array([np.eye(d)] * k)
    log_likelihoods = []

    for iteration in range(max_iter):
        # E-step
        resp = np.zeros((n, k))  # Responsibilities
        for i in range(k):
            resp[:, i] = pi[i] * multivariate_normal.pdf(data, mean=mu[i], cov=sigma[i])
        resp /= resp.sum(axis=1, keepdims=True)  # Normalize to get probabilities

        # M-step
        Nk = resp.sum(axis=0)  # Effective number of points in each cluster
        for i in range(k):
            mu[i] = (resp[:, i][:, np.newaxis] * data).sum(axis=0) / Nk[i]
            diff = data - mu[i]
            sigma[i] = (resp[:, i][:, np.newaxis] * diff).T @ diff / Nk[i]
            sigma[i] += np.eye(d) * 1e-6  # Regularization to avoid singular matrices
        pi = Nk / n  # Update mixing coefficients

        # Compute log-likelihood
        log_likelihood = np.sum(np.log(np.sum([
            pi[j] * multivariate_normal.pdf(data, mean=mu[j], cov=sigma[j])
            for j in range(k)], axis=0)))
        log_likelihoods.append(log_likelihood)

        # Check for convergence
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return mu, sigma, pi, log_likelihoods

# Run EM
mu, sigma, pi, log_likelihoods = em_gmm(data, k)

# Print results
print("Estimated means:\n", mu)
print("Estimated covariances:\n", sigma)
print("Estimated mixing coefficients:\n", pi)

# Visualization
import matplotlib.pyplot as plt

plt.scatter(data[:, 0], data[:, 1], c='gray', s=20, label="Data")
colors = ['red', 'blue']
for i in range(k):
    plt.scatter(mu[i][0], mu[i][1], c=colors[i], marker='x', s=100, label=f"Cluster {i+1} Mean")
plt.title("EM Algorithm for GMM")
plt.legend()
plt.show()
