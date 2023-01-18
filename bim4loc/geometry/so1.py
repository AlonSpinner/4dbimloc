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
    '''
    This doesnt work for now... averging rotations is not trivial

    example:

    lets say we have a lot of angles -3 and +3 scattered around 3.14
    averaging in the lie algebra will give us 0, but the average in the manifold is 3.14
    averging in the manifold in a stupid way breaks the manifold
    '''
    mu = np.full(z_array[0].shape,fill_value = exp(0))
    for i in prange(len(z_array)):
        mu = plus(mu, z_array[i])
    mu /= len(z_array)
    #try to normalize.. this doesnt work
    mu[0] = mu[0]/np.abs(mu[0])
    mu[1]= mu[1]/np.abs(mu[1])
    return mu

