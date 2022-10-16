import numpy as np
import numba as nb

def f(mu, cov, n):
    return np.random.multivariate_normal(mu, cov, n)

nbf = nb.njit(f)

print(np.random.multivariate_normal(np.array([1,2]), np.array([[1,0],[0,1]]), 10))
print(nbf(np.array([1,2]), np.array([[1,0],[0,1]]), 10))