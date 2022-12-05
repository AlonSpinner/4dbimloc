import numpy as np
from . import one_dim as r1d
from numba import njit, prange
from scipy.linalg import cholesky

@njit(parallel = True, cache = True)
def sample_uniform(a : np.ndarray, b : np.ndarray, n : int):
    m = a.shape[0]
    s = np.zeros((n,m))
    for i in prange(n):
        for j in prange(m):
            s[i,j] = np.random.uniform(a[j],b[j])
    return s

@njit(parallel = True, cache = True)
def sample_normal(mu : np.ndarray, cov : np.ndarray, n : int):
    '''
    mu - np.ndarray of shape (m,)
    cov - np.ndarray of shape (m,m)
    n - int

    returns np.ndarray of shape (n,m) of samples
    '''
    #based on https://juanitorduz.github.io/multivariate_normal/
    m = mu.shape[0]
    flat_cov = cov.flatten()
    eps = np.min(flat_cov[np.nonzero(flat_cov)]) *1e-6
    L = np.linalg.cholesky(cov + eps*np.eye(m))
    s = np.zeros((n,m))

    for i in prange(n):
        u = np.zeros(m)
        for j in prange(m):
            u[j] = np.random.normal(0,1)
        s[i] = mu + np.dot(L,u)
        
    return s

@njit(cache = True)
def gauss_likelihood(x : np.ndarray, mu : np.ndarray, cov : np.ndarray, pseudo = False):
    inExp = (-0.5*(x-mu).T @ np.linalg.inv(cov) @ (x-mu))
    num = np.exp(inExp)
    if pseudo:
        return num
    else:
        k = x.size
        den = np.sqrt((2.0 * np.pi) ** k * np.linalg.det(cov))
        return num/den

@njit(parallel = True, cache = True)
def gauss_likelihood_uncoupled(x_vec : np.ndarray, mu_vec : np.ndarray, std_vec : np.ndarray, pseudo = False):
    p = 1.0
    for mu, std, x in zip(mu_vec, std_vec, x_vec):
        p *= r1d.Gaussian._pdf(mu, std, x)
    return p

@njit(cache = True)
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

class SigmaPoints():
    def __init__(self, n : int, alpha : float = 1.0, beta : float = 2.0,
                        mu = None, cov = None):
        self.n = n
        self._alpha = alpha
        self._beta = beta
        self._kappa = 3 - n
        self._lamBda = alpha**2 * (n + self._kappa) - n
        
        self.weights = self._compute_weights()
        if mu is not None and cov is not None:
            self.points = self.compute_sigma_points(mu, cov)
    
    def _compute_weights(self):
        weights = 0.5/(self.n+self._lamBda) * np.ones(2*self.n+1)
        weights[-1] = self._lamBda/(self.n+self._lamBda) #overwrite
        return weights

    def compute_sigma_points(self, mean : np.ndarray, cov : np.ndarray) -> np.ndarray:
        M = np.sqrt(self.n+self._lamBda) * cov
        points = np.zeros((2*self.n + 1, self.n))
        for i in range(self.n):
            points[i] = mean + M[i]
            points[i+self.n] = mean - M[i]
        points[-1] = mean
        return points
