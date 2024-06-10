import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
import time
from himmelblau import himmelblau
from stretch_move import stretch_move
import tqdm

# Log-probability function for MCMC
def log_prob(theta):
    return -himmelblau(theta)

# Initialize parameters
nwalkers = 40
ndim = 2
nsteps = 100000
a = 2
p0 = 8 * np.random.rand(nwalkers, ndim) - 4 # Initial grid

# Print initial conditions
print("")
print("Comparison costum vs emcee")
print("Himmelblau function")

print("") 
print("ndim:", ndim)
print("nwalkers:", nwalkers)
print("nsteps:", nsteps)
print("Initial positions:", np.min(p0), "-", np.max(p0))

# Run the custom MCMC sampler
samples_costum, acceptance_count, delta_t_costum = stretch_move(log_prob, p0, nsteps, nwalkers, ndim, a)

# Calculate the mean acceptance rate
mean_acceptance_rate = acceptance_count / (nsteps * nwalkers)

# Calculate the autocorrelation times
autocorr_times = emcee.autocorr.integrated_time(samples_costum)

# Cut away 5 times the autocorellation time from samples
samples_costum = samples_costum[5 * int(np.max(autocorr_times)):]

# Print results
print("")
print("Costum implementation")
print("Autocorrelation times for each dimension:", autocorr_times)
print("Mean acceptance rate:", mean_acceptance_rate)
print(f"Runtime for {nsteps}:", delta_t_costum)

# Run the emcee sampler
start_time = time.time()

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, moves=emcee.moves.StretchMove(a=a))

# Use state as new initial values
sampler.run_mcmc(p0, nsteps, progress=True)
samples_emcee = sampler.get_chain()
end_time = time.time()

delta_t_emcee = end_time - start_time

autocorr_times = emcee.autocorr.integrated_time(samples_emcee)

# Cut away 5 times the autocorrelation time from samples
samples_emcee = samples_emcee[int(5 * np.max(autocorr_times)):]

# Print results for emcee
print("")
print("emcee implementation")
print("Autocorrelation times for each dimension:", autocorr_times)
print("Mean acceptance rate:", np.mean(sampler.acceptance_fraction))
print(f"Runtime for {nsteps}:", delta_t_emcee)

# Create and store the corner plots
fig = corner.corner(samples_costum.reshape(-1, ndim), bins=50, labels=["$x1$", "$x2$"])
fig.suptitle("Custom implementation")
fig.savefig("img/corner_custom_stretch_move.png")

fig = corner.corner(samples_emcee.reshape(-1, ndim), bins=50, labels=["$x1$", "$x2$"])
fig.suptitle("emcee implementation")
fig.savefig("img/corner_emcee.png")
