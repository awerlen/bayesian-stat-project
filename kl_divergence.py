import numpy as np

# Calculate the KL divergence
def kl_divergence(p, q, bins):
    p_hist, _ = np.histogramdd(p, bins=bins, density=True)
    q_hist, _ = np.histogramdd(q, bins=bins, density=True)
    p_hist = p_hist / np.sum(p_hist)
    q_hist = q_hist / np.sum(q_hist)

    mask = (p_hist > 0) & (q_hist > 0)
    kl_div = np.sum(p_hist[mask] * np.log(p_hist[mask] / q_hist[mask]))
    return kl_div
