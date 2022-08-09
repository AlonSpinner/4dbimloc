import bim4loc.random_models.one_dim as r1d
import bim4loc.sensors as sensors
import numpy as np
from typing import Literal

def inverse_existence_model(z : str, m : str) -> float:
    '''
    returns the probability of m given measurement z
    ⬛ - exists
    ⬜ - doesnt exist
    '''
    if m == "⬛":
        if z == "⬜":
            return 0.1
        elif z == "⬛":
            return 0.9

    elif m == "⬜":
        if z == "⬜":
            return 0.85
        elif z == "⬛":
            return 0.15


def m_given_rangeWorld_rangeBelief(m : Literal["⬛","⬜"], wz : np.ndarray, bz : np.ndarray, std) -> np.ndarray:
    '''
    returns the probability of m given two range measurements
    m = "⬛"- exists
    m = "⬜" - doesnt exist
    wz - world measurement
    bz - belief measurement
    std - range sensor standard deviation
    '''
    if m == "⬛":
        return r1d.Gaussian._pdf(bz, wz, std)
    elif m == "⬜":
        return 1 - r1d.Gaussian._pdf(bz, wz, std)