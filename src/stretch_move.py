import numpy as np
import tqdm
import time

def stretch_move(log_prob, p0, nsteps, nwalkers, ndim, a):
    # Stretch move for MCMC

    def next_iteration(walkers, log_probs, log_prob, a):
        nwalkers, ndim = walkers.shape
        new_walkers = np.zeros_like(walkers)
        new_log_probs = np.zeros(nwalkers)
        accepted = np.zeros(nwalkers, dtype=bool)
        
        for i in range(nwalkers):
            # Choose a random walker other than the current one
            idx = np.random.randint(nwalkers - 1)
            if idx >= i:
                idx += 1
            z = np.random.uniform(low=1.0/a, high=a)
            new_pos = walkers[idx] + z * (walkers[i] - walkers[idx])
            new_log_prob = log_prob(new_pos)
            
            # Acceptance criterion
            if np.log(np.random.rand()) < (ndim - 1) * np.log(z) + new_log_prob - log_probs[i]:
                new_walkers[i] = new_pos
                new_log_probs[i] = new_log_prob
                accepted[i] = True
            else:
                new_walkers[i] = walkers[i]
                new_log_probs[i] = log_probs[i]
        
        return new_walkers, new_log_probs, accepted
    
    # Initialize walkers and log-probabilities
    nwalkers = p0.shape[0]
    ndim = p0.shape[1]
    log_probs = np.array([log_prob(w) for w in p0])

    # Run the MCMC sampler
    samples = np.zeros((nsteps, nwalkers, ndim))
    acceptance_count = 0
    time_start = time.time()
    for step in tqdm.tqdm(range(nsteps)):
        p0, log_probs, accepted = next_iteration(p0, log_probs, log_prob, a)
        samples[step] = p0
        acceptance_count += np.sum(accepted)
    time_end = time.time()
    delta_t = time_end - time_start

    return samples, acceptance_count, delta_t

