import numpy as np
from scipy.special import logsumexp
from scipy.stats import uniform, beta
import tqdm
import scipy.stats
import matplotlib.pyplot as plt

# Define the Himmelblau function
def himmelblau(point):
    x, y = point
    return -((x**2 + y - 11)**2 + (x + y**2 - 7)**2)

# Log likelihood function for the nested sampling algorithm
def log_likelihood(points):
    return np.apply_along_axis(himmelblau, 1, points)

# Nested sampling algorithm
def sample_nested_sampling(log_likelihood, prior, n_live, tol=0.01, n_max_iter=100000):
    # Sample the initial set of live points from the prior
    live_points = np.column_stack((prior.rvs(n_live), prior.rvs(n_live)))

    # Get their log likelihoods
    log_L = log_likelihood(live_points)
    if not np.all(np.isfinite(log_L)):
        raise ValueError("Non-finite log likelihood for some points.")

    # Set up some book-keeping
    log_tol = np.log(tol)

    log_X = [0,]

    dead_points = []
    dead_points_log_L = []

    n_eval = live_points.shape[0]
    drain_live_points = False
    i = 0
    progress = tqdm.tqdm()
    while i < n_max_iter:
        # Find the live point with the lowest likelihood
        idx = np.argmin(log_L)
        # Call the likelihood at this point L^*
        log_L_star = log_L[idx]

        # This lowest likelihood point becomes a dead point
        dead_points.append(live_points[idx])
        dead_points_log_L.append(log_L_star)

        # Estimate the shrinkage of the likelihood volume when removing the
        # lowest-likelihood point
        log_t = -1 / n_live
        log_X.append(log_X[-1] + log_t)

        # Check for convergence of the evidence estimate
        if i > 4:
            # Compute the volumes and weights
            X = np.exp(np.array(log_X))
            w = 0.5 * (X[:-2] - X[2:])
            # Estimate Z = \sum_i w_i L^*_i
            log_Z = logsumexp(np.array(dead_points_log_L[:-1]), b=w)
            # Estimate the error on Z as the mean of the likelihoods of the
            # live points times the current likelihood volume
            # \Delta Z = X_i \frac{1}{n_{live}}\sum_j L_j
            log_mean_L = logsumexp(log_L, b=1/n_live)
            log_Delta_Z = log_mean_L + log_X[-1]
            # If the estimated error is less than the tolerance, stop sampling
            # new live points for the dead points that get removed
            if log_Delta_Z - log_Z < log_tol:
                drain_live_points = True
                live_points = np.delete(live_points, idx, axis=0)
                log_L = np.delete(log_L, idx)
                if len(log_L) == 0:
                    break

            progress.set_postfix({"log_Z": log_Z, "n_eval": n_eval, "iter": i})

        # Sample a new live point from the prior with a likelihood higher than
        # L^*
        while not drain_live_points:
            # Sampling from the whole prior is very inefficient, in practice
            # there are more sophisticated sampling schemes
            new_point = np.column_stack((prior.rvs(1), prior.rvs(1))).squeeze()
            log_L_new = log_likelihood(new_point[np.newaxis, :])[0]  # Ensure it's 2D for the likelihood function
            n_eval += 1
            if np.isfinite(log_L_new) and log_L_new > log_L_star:
                live_points[idx] = new_point
                log_L[idx] = log_L_new
                break

        i += 1

    # Because the estimate of the volumes is stochastic, we can sample many of 
    # them to get the uncertainty on our evidence estimate
    dead_points = np.array(dead_points)
    dead_points_log_L = np.array(dead_points_log_L)
    n_sample = 100
    t_sample = beta(n_live, 1).rvs((n_sample, len(dead_points_log_L)))
    log_X_sample = np.insert(np.cumsum(np.log(t_sample), axis=1), 0, 0, axis=1)
    X_sample = np.exp(log_X_sample)
    w_sample = 0.5 * (X_sample[:, :-2] - X_sample[:, 2:])
    log_Z = logsumexp(dead_points_log_L[:-1], b=w_sample, axis=1)

    return log_Z, dead_points, w * np.exp(dead_points_log_L)[:-1]

# Define the prior distribution (two-dimensional uniform distribution)
prior = uniform(loc=-4, scale=8)

# Call the nested sampling algorithm with the two-dimensional Himmelblau function
log_Z, dead_points, weights = sample_nested_sampling(
    log_likelihood, prior=prior, n_live=500, tol=.5, n_max_iter=10000)


weights_normalized = weights / np.sum(weights)
dead_points = dead_points[:len(weights_normalized)]

print("Dead points shape:", dead_points.shape)
print("Weights shape:", weights_normalized.shape)


plt.figure(figsize=(10, 8))
plt.scatter(dead_points[:, 0], dead_points[:, 1], s=weights_normalized * 1000, alpha=0.5)
plt.title('Posterior Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Normalized Weights')
plt.show()
