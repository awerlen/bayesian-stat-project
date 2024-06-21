import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
import time

from src.himmelblau import himmelblau
from src.rosenbrock import rosenbrock_3d
from src.stretch_move import stretch_move
import tqdm

# ---------------------------
# Comparison between costum MCMC implementation and emcee
# ---------------------------

# File name
function = "himmelblau"
path = f"results/SM/{function}"
file_name = f"{path}/{function}.txt"

# Log-probability function for MCMC
def log_prob(theta):
    return -himmelblau(theta)

# Initialize parameters for MCMC
nwalkers = 40
ndim = 2
nsteps = 100000
a = 1.5
p0 = 8 * np.random.rand(nwalkers, ndim) - 4 # Initial grid

# store output in file
f = open(file_name, "w")
f.write("Comparison costum vs emcee\n")
f.write(f"{function} function\n")
f.write("\n")
f.write("ndim: " + str(ndim) + "\n")
f.write("nwalkers: " + str(nwalkers) + "\n")
f.write("nsteps: " + str(nsteps) + "\n")
f.write("Stretch factor: " + str(a) + "\n")
f.write("Initial positions: " + str(np.min(p0)) + " - " + str(np.max(p0)) + "\n")

# Run the custom MCMC sampler
print("Run costum MCMC sampler")
samples_costum, acceptance_count, delta_t_costum_costum = stretch_move(log_prob, p0, nsteps, nwalkers, ndim, a)

# Calculate the mean acceptance rate
mean_acceptance_rate_costum = acceptance_count / (nsteps * nwalkers)

# Calculate the autocorrelation times
autocorr_times_costum = emcee.autocorr.integrated_time(samples_costum)

# Cut away 5 times the autocorellation time from samples
samples_costum_uncut = samples_costum
samples_costum = samples_costum[5 * int(np.max(autocorr_times_costum)):]

# Print results
f.write("\n")
f.write("Costum implementation \n")
f.write(f"Autocorrelation times for each dimension: {autocorr_times_costum} \n")
f.write(f"Mean acceptance rate: {mean_acceptance_rate_costum} \n")
f.write(f"Runtime for {nsteps}: {delta_t_costum_costum} \n")

# Run the emcee sampler
start_time = time.time()

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, moves=emcee.moves.StretchMove(a=a))

print("Run emcee sampler")
sampler.run_mcmc(p0, nsteps, progress=True)
samples_emcee = sampler.get_chain()
end_time = time.time()

# Calculate runtime
delta_t_emcee = end_time - start_time

# Calculate the autocorrelation times
autocorr_times = emcee.autocorr.integrated_time(samples_emcee)

# Cut away 5 times the autocorrelation time from samples
samples_emcee = samples_emcee[int(5 * np.max(autocorr_times)):]

# Print results for emcee
f.write("\n")
f.write("emcee implementation \n")
f.write(f"Autocorrelation times for each dimension: {autocorr_times} \n")
f.write(f"Mean acceptance rate: {np.mean(sampler.acceptance_fraction)} \n")
f.write(f"Runtime for {nsteps}: {delta_t_emcee} \n")
f.close()

# ---------------------------
# Plotting
# ---------------------------

# Define labels for three dimensions
labels = ["$x_1$", "$x_2$"]
fontsize = 11
# Create figure and axes for a 3x3 grid
fig, axes = plt.subplots(ndim, ndim, figsize=(5, 5))

# Reshape samples to match the 3D case
samples_custom_reshaped = samples_costum.reshape(-1, ndim)
samples_emcee_reshaped = samples_emcee.reshape(-1, ndim)

# define range for plot
range = [[-5, 5], [-5, 5]]

# Generate corner plots for both sets of samples
corner.corner(samples_custom_reshaped, bins=50, labels=labels, range=range, fig=fig, color="blue")
corner.corner(samples_emcee_reshaped, bins=50, labels=labels, range=range, fig=fig, color="red")

# Create legend handles
handles = [
    plt.Line2D([0], [0], color="blue", lw=2, label="SM (custom)"),
    plt.Line2D([0], [0], color="red", lw=2, label="SM (emcee)")
]
fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=fontsize-1)

# Adjust axes for the new grid
axes = np.array(fig.axes).reshape((ndim, ndim))

# Adjust x-axis labels
for ax in axes[-1, :]:
    ax.xaxis.label.set_fontsize(fontsize+2)
    ax.xaxis.label.set_position((0.5, -0.2))

# Adjust y-axis labels
for ax in axes[:, 0]:
    ax.yaxis.label.set_fontsize(fontsize+2)
    ax.yaxis.label.set_position((-0.25, 0.5))

# Adjust tick parameters
for ax in fig.axes:
    ax.tick_params(axis='x', direction='in', labelsize=fontsize)
    ax.tick_params(axis='y', direction='in', labelsize=fontsize)

# Ensure layout is tight and save the figure
fig.tight_layout()
fig.savefig(f"{path}/{function}_corner_comparison.pdf")