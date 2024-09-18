from math import log2

def kl_divergence(p, q, epsilon=1e-10):
    # Align the lengths of p and q by extending with epsilon
    length = max(len(p), len(q))
    p_extended = p + [epsilon] * (length - len(p))
    q_extended = q + [epsilon] * (length - len(q))
    
    # Normalize the distributions to ensure they sum to 1
    p_sum = sum(p_extended)
    q_sum = sum(q_extended)
    p_normalized = [pi / p_sum for pi in p_extended]
    q_normalized = [qi / q_sum for qi in q_extended]
    
    # Calculate KL divergence
    return sum(p_normalized[i] * log2(p_normalized[i] / q_normalized[i]) for i in range(length))

# Define distributions
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15]

# Calculate (P || Q)
kl_pq = kl_divergence(p, q)
print('KL(P || Q): %.3f bits' % kl_pq)

# Calculate (Q || P)
kl_qp = kl_divergence(q, p)
print('KL(Q || P): %.3f bits' % kl_qp)