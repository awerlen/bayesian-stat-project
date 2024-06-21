import corner
import dynesty
import matplotlib.pyplot as plt
import numpy as np
from dynesty import plotting as dyplot
from src.himmelblau import himmelblau
from src.rosenbrock import rosenbrock_3d
import time
# ---------------------------
# Nested Sampling
# ---------------------------

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

# initialize the sampler
sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim)

# run the sampler
start_time = time.time()
sampler.run_nested()
results = sampler.results
results.summary()
end_time = time.time()

delta_t = end_time - start_time

# write results to file
f.write(f"Number of iterations: {results.niter}\n")
f.write(f"Evidence: {results.logz[-1]} +/- {results.logzerr[-1]}\n")
f.write(f"Effective sample size: {results.eff}\n")
f.write(f"Run time: {delta_t} seconds\n")

# change the font size of the plots
plt.rcParams.update({'font.size': 14})

# extract the samples and weights
samples = results.samples
weights = np.exp(results.logwt - results.logz[-1])

# ---------------------------
# Plot the results
# ---------------------------

plt.rcParams.update({'font.size': 28})

# Plot a summary of the run
rfig, raxes = dyplot.runplot(results)
for ax in np.atleast_1d(raxes):  # Ensure raxes is treated as an array, even if it contains a single element
    ax.set_xlabel('-log $X$')

raxes[3].set_ylabel('Evidence $Z$')

# Adjust layout and save the figure
rfig.tight_layout()
rfig.savefig(f"{path}/{function}_run_plot.pdf")

plt.rcParams.update({'font.size': 15})

# Plot traces and 1-D marginalized posteriors.
tfig, taxes = dyplot.traceplot(results)

# Flatten the nested array structure
flat_taxes = taxes.ravel()

# Loop through each axis and set the x-axis label
for ax in flat_taxes:
    if isinstance(ax, plt.Axes):  # Ensure ax is a valid Axes object
        ax.set_xlabel('-log $X$')

tfig.tight_layout()

tfig.savefig(f"{path}/{function}_trace_plot.pdf")

# Corner plots
fig, axes = plt.subplots(ndim, ndim, figsize=(5, 5))

# cut of the first 1000 samples
samples = samples[1000:, :]
corner.corner(samples, labels=["$x_1$", "$x_2$"], fig=fig, bins=50)

# Adjust axes for the new grid
axes = np.array(fig.axes).reshape((ndim, ndim))

# Adjust tick parameters
for ax in fig.axes:
    ax.tick_params(axis='x', direction='in', labelsize=14)
    ax.tick_params(axis='y', direction='in', labelsize=14)

# Ensure layout is tight and save the figure
fig.tight_layout()
fig.savefig(f"{path}/{function}_corner.pdf")