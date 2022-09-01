from bim4loc.random.utils import p2logodds, p2odds, negate
from bim4loc.geometry.raycaster import NO_HIT
import bim4loc.random.one_dim as r_1d
import numpy as np
from numba import njit
from numba import prange

EPS = 1e-16

#extract for numba performance
gaussian_pdf = r_1d.Gaussian._pdf 
exponentialT_pdf  = r_1d.ExponentialT._pdf

@njit(cache = True)
def forward_sensor_model(wz : np.ndarray, #wrapper for Gaussian_pdf
            sz : np.ndarray, 
            std : float, 
            pseudo = True) -> np.ndarray:
    '''
    input:
    wz - world range measurement
    sz - simulated range measurement
    std - range sensor standard deviation (equivalent for both sensors)
    pseudo - if True, doesnt normalize gaussian (p ~ exp(-0.5 * (wz - sz)**2 / std**2))

    output:
    probabilty of measuring wz : p(wz|sz,m)

    in the future:
    sigma = f(sz)
    sigma = f(angle of ray hit)
    '''

    return gaussian_pdf(mu = sz, sigma = std, x =  wz, pseudo = pseudo)

@njit(cache = True)
def inverse_sensor_model(wz_i, sz_i, szid_i, beliefs, sensor_std, sensor_max_range):
    '''
    based on the awesome papers "Autonomous Exploration with Exact Inverse Sensor Models"
    and "Bayesian Occpuancy Grid Mapping via an Exact Inverse Sensor Model"
    by Evan Kaufman et al.
    
    '''
    N_maxhits = sz_i.size
    valid_hits = 0
    for j in prange(N_maxhits):
        if szid_i[j] == NO_HIT:
            break
        valid_hits += 1

    Pjbar = 1.0
    inv_eta = 0.0
    pj_z_i_wave = np.zeros(valid_hits)

    #random hit
    inv_eta = inv_eta + exponentialT_pdf(0.01 * sensor_max_range , \
                                                sensor_max_range, wz_i)

    #solids
    for j in prange(valid_hits):
        sz_ij = sz_i[j]
        belief_ij = beliefs[szid_i[j]]

        Pjplus = Pjbar * belief_ij
        Pjbar = Pjbar * negate(belief_ij)
        
        a_temp = Pjplus * forward_sensor_model(wz_i, sz_ij, sensor_std, pseudo = True)
        pj_z_i_wave[j] = belief_ij * inv_eta + a_temp
        inv_eta = inv_eta + a_temp
    
    #max range hit
    inv_eta = inv_eta + Pjbar * forward_sensor_model(wz_i, sensor_max_range, sensor_std, pseudo = True)
    
    pj_z_i = pj_z_i_wave / inv_eta
    return pj_z_i, inv_eta

@njit(cache = True)
def binary_variable_update(current, update):
    #from page 30 in Robotic Mapping and Exporation (Occpuancy Probability Mapping)    
    return np.reciprocal(1.0 + p2odds(negate(current)) * p2odds(negate(update)))

@njit(parallel = True, cache = True)
def exact(beliefs : np.ndarray, 
                  world_z : np.ndarray, 
                  simulated_z : np.ndarray, 
                  simulated_z_ids : np.ndarray,
                  sensor_std : float,
                  sensor_max_range : float) -> np.ndarray:
    '''
    inputs: 
        beliefs : one dimensional np.ndarray sorted same as solids
        world_z - range measurements from real-world-sensor 
                    np.array of shape (n_rays)
        simulated_z - range measurements from simulated sensor rays
                    np.array of shape (n_rays, max_hits)
                    if hit that was not detected, value is set to sensor_max_range
        simulated_z_ids - ids of solids that were hit from simulated rays
                    np.array of shape (n_rays, max_hits)
                    a no hit is represented by NOT_HIT constant imoprted from bim4loc.geometry.raytracer
        sensor_std - standard deviation of sensor
        sensor_max_range - max range of sensor
    
    outputs:
        beliefs - updated beliefs
    '''

    N_rays = world_z.shape[0]

    for i in prange(N_rays):
        wz_i = world_z[i]
        sz_i = simulated_z[i]
        szid_i =  simulated_z_ids[i]

        pj_zi, _ = inverse_sensor_model(wz_i, sz_i, szid_i, beliefs, sensor_std, sensor_max_range)
        for j, p in enumerate(pj_zi):
            szid_ij = szid_i[j]
            beliefs[szid_ij] = binary_variable_update(beliefs[szid_ij], p)
        
    return beliefs

@njit(parallel = True, cache = True)
def approx(logodds_beliefs : np.ndarray, 
                  world_z : np.ndarray, 
                  simulated_z : np.ndarray, 
                  simulated_z_ids : np.ndarray,
                  sensor_std : float,
                  sensor_max_range : float) -> np.ndarray:
    '''
    inputs: 
        logodds_beliefs : one dimensional np.ndarray sorted same as solids
        world_z - range measurements from real-world-sensor 
                    np.array of shape (n_rays)
        simulated_z - range measurements from simulated sensor rays
                    np.array of shape (n_rays, max_hits)
                    if hit that was not detected, value is set to sensor_max_range
        simulated_z_ids - ids of solids that were hit from simulated rays
                    np.array of shape (n_rays, max_hits)
                    a no hit is represented by NOT_HIT constant imoprted from bim4loc.geometry.raytracer
        sensor_std - standard deviation of sensor
        sensor_max_range - max range of sensor
    
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
        sz_i = simulated_z[i]
        szid_i =  simulated_z_ids[i]
        
        for j in prange(N_maxhits):
            sz_ij = sz_i[j]
            szid_ij = szid_i[j]
            if szid_ij == NO_HIT or sz_ij > sensor_max_range: # hits are sorted from close->far->NO_HIT. so nothing to do anymore
                break
            if wz_i < sz_ij - T: #wz had hit something before bzid
                break
            elif wz_i < sz_ij + T: #wz has hit bzid
                logodds_beliefs[szid_ij] += L09
            else: #wz has hit something after bzid, making us think bzid does not exist
                logodds_beliefs[szid_ij] += L01

    return logodds_beliefs

if __name__ == '__main__':
    approx(logodds_beliefs = np.array([0.5]),
                    world_z = np.array([0.5]), 
                    simulated_z = np.array([[0.5]]),
                    simulated_z_ids = np.array([[0]]),
                    sensor_std = 0.2,
                    sensor_max_range = 100.0)
    approx.parallel_diagnostics()
