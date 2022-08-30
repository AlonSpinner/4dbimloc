import numpy as np
from numba import njit

EPS = 1e-16

@njit(cache = True)
def negate(p):
    return 1.0 - p

@njit(cache = True)
def odds2p(odds):
    return odds / (1 + odds)

@njit(cache = True)
def p2odds(p):
    return p / max(1-p,EPS)

@njit(cache = True)
def p2logodds(p):
    return np.log(p2odds(p))

@njit(cache = True)
def logodds2p(l):
    return  np.exp(l) / (1 + np.exp(l))
