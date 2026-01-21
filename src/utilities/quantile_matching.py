import scipy.sparse as sp
from scipy.stats import poisson, nbinom
import numpy as np
from src.utilities.nbinom import convert_n_binom_params

def get_qm_fun(dist, dist_params = None):
    """
    Returns the quantile function for the specified distribution.
    """
    if dist_params is None:
        dist_params = {}
    if dist == "poisson":
        return quantile_match_poisson
    elif dist == "nbinom":
        return lambda k, mu, f: quantile_match_nbinom(k=k, mu=mu, r=dist_params['r'], f_=f)
    else:
        raise NotImplementedError(f"Distribution {dist} is not implemented")

def quantile_match_poisson(k: np.array, mu: np.array, f_: np.array, verbose = False) -> np.array:
    """
    Performs quantile mapping from the poisson dist param mu * f_ to the poisson dist of param mu
    # works also in vectorized form when k is the same shape as mu !
    """
    # chek that the type of k and mu are not matrices (otherwise mu * f_ gives a matrix multiplication)
    if type(mu) == np.matrix:
        raise ValueError("quantile match does not accept matrices but arrays")
    
    muf = mu * f_
    quantile = poisson.cdf(k,mu = muf)
    if verbose:
        print(f"{k=}, {muf=},{quantile=}")
    res = poisson.ppf(q = quantile, mu = mu)
    #replace where both k and mu are zeros by zeros. This is correct, and by consistency...
    zero_selector = (k == 0) & (mu == 0)
    if np.sum(zero_selector) > 0:
        res[zero_selector] = 0
    return res

def quantile_match_nbinom(k: np.array, mu: np.array, r: np.array, f_: np.array, verbose = False) -> np.array:
    """
    Performs quantile mapping from the nbinom dist param mu * f_, r to the poisson dist of param mu, r
    # works also in vectorized form when k is the same shape as mu !
    """
    # chek that the type of k and mu are not matrices (otherwise mu * f_ gives a matrix multiplication)
    if type(mu) == np.matrix:
        raise ValueError("quantile match does not accept matrices but arrays")
    
    muf = mu * f_
    n, p = convert_n_binom_params(mu = muf, alpha = r)
    quantile = nbinom.cdf(k,n = n, p = p)
    if verbose:
        print(f"{k=}, {muf=},{quantile=}")
    n, p = convert_n_binom_params(mu = mu, alpha = r)
    res = nbinom.ppf(q = quantile, n = n, p = p)
    #replace where both k and mu are zeros by zeros. This is correct, and by consistency...
    zero_selector = (k == 0) & (mu == 0)
    if np.sum(zero_selector) > 0:
        res[zero_selector] = 0
    return res


def quantile_match_sparse(k, mu, f_, verbose = False, method = quantile_match_poisson) -> sp.sparray:
    """
    Performs quantile mapping from the poisson dist param mu * f_ to the poisson dist of param mu. 
    Performs the operation on sparse matrices, which can be a lot faster in case mu is sparse. Indeed, when mu is zero, outputs zero !
    # works also in vectorized form when k is the same shape as mu !
    """
    # Ensure k and mu are sparse matrices

    f__ = np.broadcast_to(f_, mu.shape)
    assert mu.shape == k.shape
    assert mu.shape == f__.shape
    
    # potentially converts csr_matrix to csr_array
    if type(k) != sp.csr_array:
        k = sp.csr_array(k)
    if type(mu) != sp.csr_array:
        mu = sp.csr_array(mu)
        
    # Create an output matrix of the same shape as k and mu, initialized to zero
    result = sp.lil_matrix(k.shape)
    
    # Find indices where mu is non-zero. NB if mu is zero and k is not, we have a inconsistency anyway.
    indices = mu.nonzero()

    if len(indices[0]) > 0:
        quantile_match_data = method(k[indices], mu[indices], f__[indices], verbose)
        #assign_view(result[indices], quantile_match_data.reshape(result[indices].shape)) #TODO: this DOES NOT WORK !!!!
        result[indices] = quantile_match_data.reshape(result[indices].shape)

    # Return the result in a sparse format
    return result