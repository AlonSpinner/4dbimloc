import numpy as np
from numba import njit

EPS = 1e-16

@njit(cache = True)
def negate(p):
    return 1.0 - p

@njit(cache = True)
def odds2p(odds):
    return odds / (1.0 + odds)

@njit(cache = True)
def p2odds(p):
    return p /np.maximum(1.0 - p, EPS)

@njit(cache = True)
def p2logodds(p):
    return np.log(p2odds(p))

@njit(cache = True)
def logodds2p(l):
    l = np.minimum(l, 5.0)
    return  np.exp(l) / (1.0 + np.exp(l))
