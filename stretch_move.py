import numpy as np


# Stretch move for MCMC
def stretch_move(walkers, log_probs, log_prob, a=2.0):
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