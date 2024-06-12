import corner
import dynesty
import matplotlib.pyplot as plt
import numpy as np
from dynesty import plotting as dyplot
from src.himmelblau import himmelblau
from src.rosenbrock import rosenbrock_3d

function = "himmelblau"
path = f"results/NS/{function}"

file_name = f"{path}/{function}.txt"

# define the prior transform function
def prior_transform(uv):
    return 8 * uv - 4

# define the log-likelihood function
def log_likelihood(x):
    return -himmelblau(x)

ndim = 2

# define the number of living points used
# define the number of dimensions
f = open(file_name, "w")
f.write("Nested Sampling\n")
f.write(f"{function} function\n")
f.write("\n")
f.write(f"ndim: {ndim}\n")
f.write("\n")

# define the sampler
sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim)

# run the sampler
sampler.run_nested()
results = sampler.results
results.summary()

f.write(f"Number of iterations: {results.niter}\n")
f.write(f"Number of posterior samples: {len(results.samples)}\n")
f.write(f"Evidence: {results.logz[-1]} +/- {results.logzerr[-1]}\n")
f.write(f"Effective sample size: {results.eff}\n")

plt.rcParams.update({'font.size': 14})

# extract the samples and weights
samples = results.samples
weights = np.exp(results.logwt - results.logz[-1])

# Plot a summary of the run.
rfig, raxes = dyplot.runplot(results)
rfig.tight_layout()
rfig.savefig(f"{path}/{function}_run_plot.pdf")

# Plot traces and 1-D marginalized posteriors.
tfig, taxes = dyplot.traceplot(results)
tfig.tight_layout()
tfig.savefig(f"{path}/{function}_trace_plot.pdf")

# Do Corner plot of results
# Define labels for three dimensions
labels = ["$x_1$", "$x_2$"]

# Create figure and axes for a 3x3 grid
fig, axes = plt.subplots(ndim, ndim, figsize=(5, 5))
# cut of the first 100 samples
samples = samples[1000:, :]
corner.corner(samples, labels=labels, fig=fig, bins=50)

# Adjust axes for the new grid
axes = np.array(fig.axes).reshape((ndim, ndim))

# Adjust tick parameters
for ax in fig.axes:
    ax.tick_params(axis='x', direction='in', labelsize=14)
    ax.tick_params(axis='y', direction='in', labelsize=14)

# Ensure layout is tight and save the figure
fig.tight_layout()
fig.savefig(f"{path}/{function}_corner.pdf")