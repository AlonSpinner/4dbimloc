
from bim4loc.maps import Map
from bim4loc.random.utils import p2logodds, logodds2p
from bim4loc.geometry.raycaster import NO_HIT
import bim4loc.random.one_dim as r1d
from typing import Literal
import numpy as np
from numba import njit
from numba import prange

@njit(parallel = True, cache = True)
def forward_measurement_model(wz : np.ndarray, 
                              bz : np.ndarray, 
                              std, 
                              pseudo = True) -> np.ndarray:
    '''
    input:
    wz - world range measurement
    bz - belief range measurement
    std - range sensor standard deviation (equivalent for both sensors)
    pseudo - if True, doesnt normalize gaussian (p ~ exp(-0.5 * (wz - bz)**2 / std**2))

    output:
    probabilty of measuring wz : p(wz|bz,m)

    in the future:
    sigma = f(bz)
    sigma = f(angle of ray hit)
    '''

    return r1d.Gaussian._pdf(mu = bz, sigma = std, x =  wz, pseudo = pseudo)

def f(world_z, belief_z, sensor_std, pm):
    '''
    inputs:
    world_z - single range measurement from real-world-sensor
    belief_z - single range measurement from simulated sensor
    sensor_std - both of real and simulated senosr (must be equivalent)
    pm - belief_m - probablity that solid m does exist
    
    outputs:
    p(m|z) = p(z|m)p(m) / p(z)
    
    Algorithm:
    p(z|m) ~ N(bz,sigma)
    p(m) ~ prior
    p(z) = p(z|m)p(m)+ p(z|~m)p(~m) (via marginalization)    
    '''

    pzm = r1d.Gaussian._pdf(mu = belief_z, sigma = sensor_std, x = world_z)
    pm 
    
    pmz = pzm * pm / (pzm * pm + (1 - pzm) * (1 - pm))

@njit(parallel = True, cache = True)
def vanila_inverse(logodds_beliefs : np.ndarray, 
                  world_z : np.ndarray, 
                  simulated_z : np.ndarray, 
                  simulated_z_ids : np.ndarray,
                  sensor_std : float,
                  sensor_max_range : float) -> np.ndarray:
    '''
    inputs: 
    
        logodds_beliefs : one dimensional np.ndarray where indcies are iguids of solids
        world_z - range measurements from real-world-sensor 
                    np.array of shape (n_rays)
        simulated_z - range measurements from simulated sensor rays
                    np.array of shape (n_rays, max_hits)
                    if hit that was not detected, value is set to sensor_max_range
        simulated_z_ids - iguids of solids that were hit from simulated rays
                    np.array of shape (n_rays, max_hits)
                    a no hit is represented by NOT_HIT constant imoprted from bim4loc.geometry.raytracer
    
    outputs:
        logodds_beliefs - updated logodds_beliefs
    '''
    N_rays = world_z.shape[0]
    N_maxhits = simulated_z.shape[1]

    T = 3 * sensor_std
    L09 = p2logodds(0.9)
    L01 = p2logodds(0.1)

    for i in prange(N_rays):
        wz_i = world_z[i]
        bz_i = simulated_z[i]
        bzid_i =  simulated_z_ids[i]
        
        for j in prange(N_maxhits):
            bz_ij = bz_i[j]
            bzid_ij = bzid_i[j]
            if bzid_ij == NO_HIT: # hits are sorted from close->far->NO_HIT. so nothing to do anymore
                break
            if wz_i < bz_ij - T: #wz had hit something before bzid
                break
            elif wz_i < bz_ij + T: #wz has hit bzid
                logodds_beliefs[bzid_ij] += L09
            else: #wz has hit something after bzid, making us think bzid does not exist
                logodds_beliefs[bzid_ij] += L01

if __name__ == '__main__':
    vanila_inverse(logodds_beliefs = np.array([0.5]),
                    world_z = np.array([0.5]), 
                    simulated_z = np.array([[0.5]]),
                    simulated_z_ids = np.array([[0]]),
                    sensor_std = 0.2,
                    sensor_max_range = 100.0)
    vanila_inverse.parallel_diagnostics()
