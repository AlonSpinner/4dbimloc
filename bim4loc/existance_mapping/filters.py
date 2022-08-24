
from bim4loc.maps import Map
from bim4loc.random.utils import p2logodds, logodds2p
from bim4loc.geometry.raytracer import NO_HIT
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

def vanila_filter(m : Map, 
                  world_z : np.ndarray, 
                  belief_z : np.ndarray, 
                  belief_z_ids : np.ndarray,
                  sensor_std : float,
                  sensor_max_range : float,) -> None:
    '''
    world_z - range measurements from real-world-sensor 
                np.array of shape (n_rays)
    belief_z - range measurements from simulated sensor rays
                np.array of shape (n_rays, max_hits)
                if hit that was not detected, value is set to sensor_max_range
    belief_z_ids - iguids of solids that were hit from simulated rays
                np.array of shape (n_rays, max_hits)
                a no hit is represented by NOT_HIT constant imoprted from bim4loc.geometry.raytracer       
    '''

    
    for wz, bz, bzid in zip(world_z, belief_z, belief_z_ids):
        for bzi,bzidi in zip(bz, bzid):
            if bzidi == NO_HIT:
                continue
            if bzi < (wz - 1*sensor_std): #free
                p = logodds2p(m.solids[bzidi].logOdds_existence_belief + p2logodds(0.1))
            elif abs(wz-bzi) < 1*sensor_std:
                p = logodds2p(m.solids[bzidi].logOdds_existence_belief + p2logodds(0.9))
            else: #wz + 3*sensor_std < bz, cant say nothing about bz
                continue
        # p = logodds2p(m.solids[bsn].logOdds_existence_belief + p2logodds(inverse_existence_model("⬛", wz, bz, sensor_std, pseudo = True)))

            m.solids[bzidi].set_existance_belief_and_shader(p)