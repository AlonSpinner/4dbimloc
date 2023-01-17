import numpy as np
from numba import njit, prange

@njit(cache = True)
def plus(R1,R2):
    '''
    R1 + R2
    '''
    return R1 @ R2

@njit(cache = True)
def minus(R1,R2):
    '''
    R2 - R1
    '''
    return R1.T @ R2

@njit(cache = True)
def hat(theta):
    return np.array([[0,-theta[2],theta[1]],
                     [theta[2],0,-theta[0]],
                     [-theta[1],theta[0],0]])
@njit(cache = True)
def vee(Q):
    return np.array([Q[2,1],Q[0,2],Q[1,0]])

@njit(cache = True)
def log(R):
    t = np.arccos((np.trace(R)-1)/2)
    theta = t * vee(R - R.T)/(2*np.sin(t))
    return theta

@njit(cache = True)
def exp(theta):
    t = np.linalg.norm(theta)
    if t < 1e-10:
        return np.eye(3)
    else:
        hat_theta = hat(theta)
        return np.eye(3) + np.sin(t)/t * hat_theta + (1-np.cos(t))/(t**2) * hat_theta @ hat_theta

@njit(cache = True)
def mu_rotations(rotation_list : np.ndarray):
    mu = np.zeros(3)
    for i in prange(len(rotation_list)):
        mu += log(rotation_list[i])
    mu /= len(rotation_list)
    return exp(mu)

