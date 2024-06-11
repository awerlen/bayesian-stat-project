import corner
import dynesty
import matplotlib.pyplot as plt
import numpy as np
from dynesty import plotting as dyplot
from himmelblau import himmelblau
from rosenbrock import rosenbrock_3d

function = "rosenbrock_3d"
path = f"results/NS/{function}"

file_name = f"{path}/{function}.txt"

# define the prior transform function
def prior_transform(uv):
    return 8.0 * uv - 4.0

# define the log-likelihood function
def log_likelihood(x):
    return -rosenbrock_3d(x)

ndim = 3

# define the number of living points used
n_live_points = 500

# define the number of dimensions
f = open(file_name, "w")
f.write("Nested Sampling\n")
f.write(f"{function} function\n")
f.write("\n")
f.write(f"ndim: {ndim}\n")
f.write(f"n_live_points: {n_live_points}\n")
f.write("\n")

# define the sampler
sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim, n_live_points)

# run the sampler
sampler.run_nested()
results = sampler.results
results.summary()

f.write(f"Number of iterations: {results.niter}\n")
f.write(f"Number of posterior samples: {len(results.samples)}\n")
f.write(f"Evidence: {results.logz[-1]}\n")
f.write(f"Effective sample size: {results.eff}\n")

# extract the samples and weights
samples = results.samples
weights = np.exp(results.logwt - results.logz[-1])

# Plot the corner plot
fig = corner.corner(samples, labels=['$x_1$', '$x_2$','$x_3$'])
fig.savefig(f"{path}/{function}_corner_plot.png")

# Plot a summary of the run.
rfig, raxes = dyplot.runplot(results)
rfig.savefig(f"{path}/{function}_run_plot.png")

# Plot traces and 1-D marginalized posteriors.
tfig, taxes = dyplot.traceplot(results)
tfig.savefig(f"{path}/{function}_trace_plot.png")

# Plot the 2-D marginalized posteriors.
cfig, caxes = dyplot.cornerplot(results)
cfig.savefig(f"{path}/{function}_corner_plot.png")
