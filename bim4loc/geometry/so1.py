import numpy as np
from numba import njit, prange

@njit(cache = True)
def plus(z1,z2):
    '''
    z1 + z2
    '''
    return z2 * z1

@njit(cache = True)
def minus(z1,z2):
    '''
    z1 - z2
    '''
    return np.conjugate(z2) * z1

@njit(cache = True)
def log(z : np.ndarray):
    return np.angle(z)

@njit(cache = True)
def exp(theta : np.ndarray):
    return np.exp(1j * theta)

@njit(cache = True)
def mu_rotations(z_array : np.ndarray):
    mu = np.full(z_array[0].shape,fill_value = exp(0))
    for i in prange(len(z_array)):
        mu = plus(mu, z_array[i])
    mu /= len(z_array)
    return mu

