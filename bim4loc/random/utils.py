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
    p_bar = negate(p)
    indicies = np.argwhere(p_bar < EPS)
    p_bar[indicies] = EPS
    return p / p_bar

@njit(cache = True)
def p2logodds(p):
    return np.log(p2odds(p))

@njit(cache = True)
def logodds2p(l):
    l[l > 5.0] = 5.0 #big enough
    return  np.exp(l) / (1 + np.exp(l))
