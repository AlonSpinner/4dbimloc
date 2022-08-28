import numpy as np
from typing import Union
from numba import njit

EPS = 1e-16


@njit(cache = True)
def p2logodds(p) -> Union[float,np.ndarray]:
    return np.log(p / max(1 - p,EPS))

@njit(cache = True)
def logodds2p(l) -> Union[float,np.ndarray]:
    return  np.exp(l) / (1 + np.exp(l))

@njit(cache = True)
def odds2p(odds):
    return odds / (1 + odds)

@njit(cache = True)
def p2odds(p):
    return p / max(1-p,EPS)
