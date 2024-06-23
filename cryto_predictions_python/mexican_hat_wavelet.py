
import pywt
import numpy as np

wavelist = pywt.wavelist(kind='continuous')
print('wavelist', wavelist)

# wavelist = pywt.wavelist(kind='discrete')
# print('wavelist', wavelist)

wavelet = pywt.Wavelet('db2')

[dec_lo, dec_hi, rec_lo, rec_hi] = wavelet.filter_bank

print('dec_lo', dec_lo)
print('dec_hi', dec_hi)
print('rec_lo', dec_lo)
print('rec_hi', dec_hi)


scales = np.arange(1, 3)

# sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7, 8, 9])
# print('data', sig)
# ca, cd = pywt.dwt(sig, 'db3', mode)
# print('ca', ca)
# print('cd', cd)


# sig = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8, 9]])
# print('data', sig)
# ca, cd = pywt.dwt2(sig, 'db1', mode)
# print('ca', ca)
# print('cd', cd)

w_type = 'db2'
mode = 'periodization'
sig = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])

wavelet = pywt.dwt2(sig, w_type, mode)
print('wavelet', w_type, wavelet)


sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
padded = pywt.pad(sig, (10, 10), 'periodization')
print('padded', padded)

import tiktoken

# Initialize the tokenizer with a specific encoding (e.g., 'gpt2')
tokenizer = tiktoken.get_encoding("gpt2")

# Example text
text = "Hello, world!!!"

# Tokenize the text
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)

# Decode the token IDs back into text
decoded_text = tokenizer.decode(token_ids)
print("Decoded Text:", decoded_text)

# Decode each token ID to see individual tokens
tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
print("Individual Tokens:", tokens)

import numpy as np
from scipy.stats import norm

# Data
X = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 6.5, 7.0, 7.5, 8.0])

print('mean', X.mean())

# Initialize parameters
pi = np.array([0.5, 0.5])
mu = np.array([1.0, 4.0])
sigma = np.array([1.0, 1.0])
tol = 1e-6
max_iter = 100

def e_step(X, pi, mu, sigma):
    N = len(X)
    K = len(pi)
    responsibilities = np.zeros((N, K))
    
    for k in range(K):
        responsibilities[:, k] = pi[k] * norm.pdf(X, mu[k], np.sqrt(sigma[k]))
    
    sum_responsibilities = responsibilities.sum(axis=1, keepdims=True)
    responsibilities /= sum_responsibilities
    return responsibilities

def m_step(X, responsibilities):
    N, K = responsibilities.shape
    N_k = responsibilities.sum(axis=0)
    
    pi_new = N_k / N
    mu_new = np.dot(responsibilities.T, X) / N_k
    sigma_new = np.zeros(K)
    
    for k in range(K):
        diff = X - mu_new[k]
        sigma_new[k] = np.dot(responsibilities[:, k], diff ** 2) / N_k[k]
    
    return pi_new, mu_new, sigma_new

def log_likelihood(X, pi, mu, sigma):
    N = len(X)
    K = len(pi)
    log_likelihood = 0
    
    for i in range(N):
        temp = 0
        for k in range(K):
            temp += pi[k] * norm.pdf(X[i], mu[k], np.sqrt(sigma[k]))
        log_likelihood += np.log(temp)
    
    return log_likelihood

# EM algorithm
log_likelihoods = []
for iteration in range(max_iter):
    responsibilities = e_step(X, pi, mu, sigma)
    pi, mu, sigma = m_step(X, responsibilities)
    
    log_likelihood_value = log_likelihood(X, pi, mu, sigma)
    log_likelihoods.append(log_likelihood_value)
    
    if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
        break

print("Final parameters:")
print(f"Mixing coefficients: {pi}")
print(f"Means: {mu}")
print(f"Variances: {sigma}")
