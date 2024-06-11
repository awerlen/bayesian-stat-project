import corner
import dynesty
import matplotlib.pyplot as plt
import numpy as np
from dynesty import plotting as dyplot
from himmelblau as himmelblau

# define the prior transform function
def prior_transform(uv):
    return 8.0 * uv - 4.0

# define the log-likelihood function
def log_likelihood(x):
    return -himmelblau(x)

# define the number of dimensions
ndim = 2

# define the number of living points used
n_live_points = 500

# define the sampler
sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim, n_live_points)

# run the sampler
sampler.run_nested()
results = sampler.results
results.summary()

# extract the samples and weights
samples = results.samples
weights = np.exp(results.logwt - results.logz[-1])

# plot the results
plt.figure(figsize=(10, 6))
plt.scatter(samples[:, 0], samples[:, 1], c=weights, cmap='magma')
plt.colorbar(label='Weight')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Nested sampling of the Himmelblau function')
plt.show()

# Plot the corner plot
corner.corner(samples, weights=weights, labels=['x', 'y'], show_titles=True, title_fmt='.2f')
plt.show()

# Plot a summary of the run.
rfig, raxes = dyplot.runplot(results)

# Plot traces and 1-D marginalized posteriors.
tfig, taxes = dyplot.traceplot(results)

# Plot the 2-D marginalized posteriors.
cfig, caxes = dyplot.cornerplot(results)

plt.show()
