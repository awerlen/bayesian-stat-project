import numpy as np
import dynesty
import matplotlib.pyplot as plt
import corner

# Define the Himmelblau function
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
# Define the prior transform function
def prior_transform(uv):
    return 10.0 * uv - 4.0  # Transform [0, 1]^2 to [-5, 5]^2

# Define the log-likelihood function
def log_likelihood(x):
    return -himmelblau(x)  # Since we are minimizing the Himmelblau function

# Set up the nested sampler
ndim = 2  # Number of dimensions
sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim)

# Run the sampler
sampler.run_nested()
results = sampler.results

results.summary()
# Extract the samples and weights
samples = results.samples
weights = np.exp(results.logwt - results.logz[-1])

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(samples[:, 0], samples[:, 1], c=weights, cmap='viridis')
plt.colorbar(label='Weight')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Nested Sampling of the Himmelblau Function')
plt.show()

# Plot the corner plot
corner.corner(samples, weights=weights, labels=['x', 'y'], show_titles=True, title_fmt='.2f')
plt.show()
