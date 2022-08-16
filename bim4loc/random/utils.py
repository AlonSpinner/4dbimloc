import numpy as np
from typing import Union

EPS = 1e-16

def p2logodds(p) -> Union[float,np.ndarray]:
    return np.log(p / max(1 - p,EPS))

def logodds2p(l) -> Union[float,np.ndarray]:
    return  np.exp(l) / (1 + np.exp(l))

def odds2p(odds):
    return odds / (1 + odds)

def p2odds(p):
    return p / max(1-p,EPS)
