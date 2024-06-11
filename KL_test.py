import numpy as np
import dynesty
import matplotlib.pyplot as plt
import corner
from scipy.stats import gaussian_kde


# Define the Himmelblau function
def himmelblau(params):
    x, y = params
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


# Define the prior transform function
def prior_transform(uv):
    return 10.0 * uv - 5.0  # Transform [0, 1]^2 to [-5, 5]^2


# Define the log-likelihood function
def log_likelihood(params):
    return -himmelblau(params)  # Since we are minimizing the Himmelblau function


# Set up the nested sampler
ndim = 2  # Number of dimensions
n_live_points = 500  # Number of living points

sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim, nlive=n_live_points)

# Run the sampler
sampler.run_nested()
results = sampler.results

# Extract the samples and weights
samples = results.samples
weights = np.exp(results.logwt - results.logz[-1])

# Plot the corner plot for nested sampling
corner.corner(samples, weights=weights, labels=['x', 'y'], show_titles=True, title_fmt='.2f')
plt.suptitle('Nested Sampling Results')
plt.show()


# Calculate the KL divergence
def kl_divergence(p, q, bins):
    p_hist, _ = np.histogramdd(p, bins=bins, density=True)
    q_hist, _ = np.histogramdd(q, bins=bins, density=True)
    p_hist = p_hist / np.sum(p_hist)
    q_hist = q_hist / np.sum(q_hist)

    mask = (p_hist > 0) & (q_hist > 0)
    kl_div = np.sum(p_hist[mask] * np.log(p_hist[mask] / q_hist[mask]))
    return kl_div


# Generate a reference distribution using MCMC as an example
import emcee


# Define the log-posterior function
def log_posterior(params):
    lp = 0.0  # Flat prior
    ll = log_likelihood(params)
    return lp + ll


# Initialize the walkers in a small Gaussian ball around the origin
nwalkers = 50
nsteps = 5000
initial_position = np.random.randn(nwalkers, ndim)

# Set up the sampler
sampler_mcmc = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

# Run the MCMC chain
sampler_mcmc.run_mcmc(initial_position, nsteps, progress=True)
samples_mcmc = sampler_mcmc.get_chain(discard=100, thin=15, flat=True)

# Calculate KL divergence between the nested sampling and MCMC results
bins = 50  # Number of bins for the histogram
kl_div = kl_divergence(samples, samples_mcmc, bins=bins)
print(f'KL Divergence: {kl_div:.4f}')

# Plot the corner plot for MCMC
corner.corner(samples_mcmc, labels=['x', 'y'], show_titles=True, title_fmt='.2f')
plt.suptitle('MCMC Results')
plt.show()