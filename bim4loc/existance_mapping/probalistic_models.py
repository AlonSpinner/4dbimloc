import bim4loc.random.one_dim as r1d
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


