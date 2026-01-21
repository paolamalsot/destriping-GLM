def convert_n_binom_params(mu, alpha):
    # go from a formulation A of negative binomial in terms of mu, alpha
    # to form. B with n, p
    # formulation B: models number of FAILURES before a given number of successes (n) occur
    # p is the success probability
    # formulation A: models a negative binomial distribution where the mean is mu, and variance = mu + alpha mu**2
    # mu is the mean, alpha is the dispersion parameter
    n = 1 / alpha
    p = n / (n + mu)
    return n, p
