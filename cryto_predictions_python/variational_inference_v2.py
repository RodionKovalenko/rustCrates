import random
import math

def generate_synthetic_data(num_samples, true_slope, true_intercept, true_noise_std):
    X = []
    y = []
    for _ in range(num_samples):
        x_i = random.gauss(0, 1)
        noise = random.gauss(0, true_noise_std)
        y_i = true_slope * x_i + true_intercept + noise
        X.append([x_i])
        y.append([y_i])
    return X, y

def matrix_multiply(A, B):
    return [[sum(a*b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

def transpose_matrix(A):
    return [list(i) for i in zip(*A)]

def inverse_matrix(matrix):
    n = len(matrix)
    AM = [[matrix[i][j] for j in range(n)] for i in range(n)]
    IM = [[float(i == j) for i in range(n)] for j in range(n)]
    for fd in range(n):
        fdScaler = 1.0 / AM[fd][fd]
        for j in range(n):
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler
        for i in range(n):
            if i == fd:
                continue
            crScaler = AM[i][fd]
            for j in range(n):
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                IM[i][j] = IM[i][j] - crScaler * IM[fd][j]
    return IM

def bayesian_linear_regression(X, y, prior_mean, prior_precision, num_iterations, tol=1e-6):
    n = len(X)
    d = len(X[0])
    posterior_mean = [[0.0] for _ in range(d)]
    posterior_precision = prior_precision

    identity = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]

    for iter in range(num_iterations):
        previous_posterior_mean = [row[:] for row in posterior_mean]

        XT = transpose_matrix(X)
        XTX = matrix_multiply(XT, X)
        XTy = matrix_multiply(XT, y)

        cov_inv = [[identity[i][j] + posterior_precision * XTX[i][j] for j in range(d)] for i in range(d)]
        cov = inverse_matrix(cov_inv)
        posterior_mean = matrix_multiply(cov, XTy)

        max_change = max(abs(posterior_mean[i][0] - previous_posterior_mean[i][0]) for i in range(d))
        if max_change < tol:
            print(f"Converged after {iter + 1} iterations.")
            break

    return posterior_mean, posterior_precision

def predict(x, posterior_mean, posterior_precision):
    mean_prediction = sum(x_i * pm[0] for x_i, pm in zip(x, posterior_mean))
    variance_prediction = 1.0 / posterior_precision + sum(x_i * pm[0] for x_i, pm in zip(x, posterior_mean))
    return mean_prediction, variance_prediction

# Parameters for synthetic data
num_samples = 100
true_slope = 2.0
true_intercept = 1.0
true_noise_std = 0.5

X, y = generate_synthetic_data(num_samples, true_slope, true_intercept, true_noise_std)

# Prior parameters
prior_mean = 0.0
prior_precision = 1.0

# Fit model using variational inference
num_iterations = 100
posterior_mean, posterior_precision = bayesian_linear_regression(X, y, prior_mean, prior_precision, num_iterations)

# Make predictions
x_new = [1.0]  # New data point to predict
mean_prediction, variance_prediction = predict(x_new, posterior_mean, posterior_precision)

print(f"Posterior mean: {mean_prediction}")
print(f"Posterior variance: {variance_prediction}")

# Validation
# X_valid [[1], [2], [3]]
# posterior_mean = [[1], [2], [3]]
def validate_model(X_valid, y_valid, posterior_mean, posterior_precision):
    # x_i, pm has a form [(1.0, [0.5]), (2.0, [1.5]), (3.0, [2.5])] in zip(x_posterior_mean)
    y_pred_mean = [sum(x_i * pm[0] for x_i, pm in zip(x, posterior_mean)) for x in X_valid]
    mse = sum((y_p - y_a[0]) ** 2 for y_p, y_a in zip(y_pred_mean, y_valid)) / len(y_valid)
    return mse

# Generate validation data
X_valid, y_valid = generate_synthetic_data(20, true_slope, true_intercept, true_noise_std)

print(X_valid)
mse = validate_model(X_valid, y_valid, posterior_mean, posterior_precision)
print(f"Validation MSE: {mse}")
