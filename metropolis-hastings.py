import numpy as np
import scipy.stats as stats

def sample_metropolis_hastings(n, x0, target_distr, sample_transition, transition_prob):
    x0 = np.atleast_1d(x0)
    samples = []
    n_rejected = 0
    n_accepted = 0
    for i in range(n):
        # Sample proposal
        x1 = sample_transition(x0)
        # Compute probabilities of the old and proposed states
        p0 = target_distr.pdf(x0)
        p1 = target_distr.pdf(x1)

        # Compute the transition probabilities
        q01 = transition_prob(x0, x1)
        q10 = transition_prob(x1, x0)

        a = p1 / p0 * q01 / q10

        u = np.random.uniform(size=1)
        if a >= u:
            # accept, proposed state becomes new state
            n_accepted += 1
            x0 = x1
            samples.append(x1)
        else:
            # reject, stay with current state
            n_rejected += 1
            samples.append(x0)

    acceptance_rate = n_accepted / (n_accepted + n_rejected)
    samples = np.array(samples)

    # Calculate autocorrelation time
    def autocorrelation(x):
        n = len(x)
        mean = np.mean(x)
        var = np.var(x)
        x = x - mean
        result = np.correlate(x, x, mode='full') / (var * n)
        return result[n-1:]

    acf = autocorrelation(samples.flatten())
    autocorr_time = 1 + 2 * np.sum(acf[1:])

    print(f"Acceptance rate: {acceptance_rate}")
    print(f"Autocorrelation time: {autocorr_time}")

    return samples