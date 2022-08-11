import numpy as np
from . import one_dim as r1d
from numba import njit

@njit(parallel = True)
def gauss_likelihood(x : np.ndarray, mu : np.ndarray, cov : np.ndarray, pseudo = False):
    inExp = (-0.5*(x-mu).T @ np.linalg.inv(cov) @ (x-mu))[0,0] #[0,0] is to get the scalar
    num = np.exp(inExp)
    if pseudo:
        return num
    else:
        k = x.size
        den = np.sqrt((2.0 * np.pi) ** k * np.linalg.det(cov))
        return num/den

@njit(parallel = True)
def gauss_likelihood_uncoupled(x_vec : np.ndarray, mu_vec : np.ndarray, std_vec : np.ndarray, pseudo = False):
    p = 1.0
    for mu, std, x in zip(mu_vec, std_vec, x_vec):
        p *= r1d.Gaussian._pdf(mu, std, x)
    return p

@njit(parallel = True)
def gauss_fit(x : np.ndarray ,p: np.ndarray):
    # x - [m,n] 
    #   n is the number of observations
    #   m is the state size
    # p - probability vector

    m = x.shape[0]
    n = x.shape[1]

    #state expactancy E(x)
    mu = x @ p

    #compute covariance E(x-E(x) @ (x-E(x).T) )
    cov = np.zeros((m,m))
    dx = x-mu.reshape(-1,1)
    dx = dx.reshape(-1,m,1)
    for i in range(n):
        cov += dx[i] @ dx[i].T * p[i]

    return mu, cov