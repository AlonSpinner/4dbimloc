
from bim4loc.maps import Map
from bim4loc.random.utils import p2logodds, logodds2p
import bim4loc.random.one_dim as r_1d
from typing import Literal
import numpy as np

def inverse_existence_model(m : Literal["⬛","⬜"], 
                            wz : np.ndarray, 
                            bz : np.ndarray, 
                            std, 
                            pseudo = True) -> np.ndarray:
    '''
    returns the probability of m given two range measurements
    m = "⬛"- exists
    m = "⬜" - doesnt exist
    wz - world measurement
    bz - belief measurement
    std - range sensor standard deviation
    '''
    if m == "⬛":
        return r_1d.Gaussian._pdf(mu = bz, sigma = std, x =  wz, pseudo = pseudo)
    elif m == "⬜":
        return 1 - r_1d.Gaussian._pdf(mu = bz, sigma = std, x =  wz, pseudo = pseudo)

def vanila_filter(m : Map, world_z : np.ndarray, belief_z : np.ndarray, sensor_std : float, belief_solid_names) -> None:

     for wz, bz, bsn in zip(world_z, belief_z, belief_solid_names):
        # if abs(wz-bz) < 3*sensor_std:
        #     p = logodds2p(m.solids[bsn].logOdds_existence_belief + p2logodds(0.9))
        # else:
        #     p = logodds2p(m.solids[bsn].logOdds_existence_belief + p2logodds(0.1))
        p = logodds2p(m.solids[bsn].logOdds_existence_belief + p2logodds(inverse_existence_model("⬛", wz, bz, sensor_std, pseudo = True)))
        
        m.solids[bsn].set_existance_belief_and_shader(p)