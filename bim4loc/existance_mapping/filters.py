from bim4loc.random.utils import p2logodds, logodds2p
from bim4loc.geometry.raycaster import NO_HIT
import bim4loc.random.one_dim as r_1d
import numpy as np
from numba import njit
from numba import prange

EPS = 1e-16

@njit(cache = True)
def binary_variable_update(prior, update):
    #from page 30 in Robotic Mapping and Exporation (Occpuancy Probability Mapping)
    return np.reciprocal(1.0 + prior/max((1.0 - prior),EPS) * (1.0 - update)/max(update,EPS))


gaussian_pdf = r_1d.Gaussian._pdf #extract for numba performance
@njit(cache = True)
def forward(wz : np.ndarray, #wrapper for Gaussian_pdf
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

@njit(parallel = True, cache = True)
def compute_p_ij(wz_i, sz_i, szid_i, beliefs, sensor_std):
    '''
    inputs:
    world_z - single range measurement from real-world-sensor
    belief_z - np.ndarray
    sensor_std - both of real and simulated senosr (must be equivalent)
    pm - belief_m - probablity that solid m does exist
    '''
    N_maxhits = sz_i.size

    p_ij = np.zeros_like(sz_i)
    pierced_meshes_probability = 1.0
    for j in prange(N_maxhits):
        sz_ij = sz_i[j]
        szid_ij = szid_i[j]
        if szid_ij == NO_HIT: # hits are sorted from close->far->NO_HIT. so nothing to do anymore
            break

        pm_ij = beliefs[szid_ij]

        w_ij = pierced_meshes_probability * pm_ij
        f_ij = forward(wz_i, sz_ij, sensor_std, pseudo = True)

        p_ij[j] = w_ij * f_ij

        #update pierced meshes probability for next iteration
        pierced_meshes_probability = pierced_meshes_probability * (1.0 - pm_ij)
    
    p_i = np.sum(p_ij) + 0.1
    p_ij = p_ij/p_i
    
    return p_ij, p_i

@njit(parallel = True, cache = True)
def vanila_forward(beliefs : np.ndarray, 
                  world_z : np.ndarray, 
                  simulated_z : np.ndarray, 
                  simulated_z_ids : np.ndarray,
                  sensor_std : float,
                  sensor_max_range : float) -> np.ndarray:

    N_rays = world_z.shape[0]
    N_maxhits = simulated_z.shape[1]

    for i in prange(N_rays):
        wz_i = world_z[i]
        sz_i = simulated_z[i]
        szid_i =  simulated_z_ids[i]
        p_ij, p_i = compute_p_ij(wz_i, sz_i, szid_i, beliefs, sensor_std)

        for j in prange(N_maxhits):
            if szid_i[j] == NO_HIT:
                break
            szid_ij = szid_i[j]
            beliefs[szid_ij] = binary_variable_update(beliefs[szid_ij],p_ij[j])


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
        sz_i = simulated_z[i]
        szid_i =  simulated_z_ids[i]
        
        for j in prange(N_maxhits):
            sz_ij = sz_i[j]
            szid_ij = szid_i[j]
            if szid_ij == NO_HIT: # hits are sorted from close->far->NO_HIT. so nothing to do anymore
                break
            if wz_i < sz_ij - T: #wz had hit something before bzid
                break
            elif wz_i < sz_ij + T: #wz has hit bzid
                logodds_beliefs[szid_ij] += L09
            else: #wz has hit something after bzid, making us think bzid does not exist
                logodds_beliefs[szid_ij] += L01

    return logodds_beliefs

if __name__ == '__main__':
    vanila_inverse(logodds_beliefs = np.array([0.5]),
                    world_z = np.array([0.5]), 
                    simulated_z = np.array([[0.5]]),
                    simulated_z_ids = np.array([[0]]),
                    sensor_std = 0.2,
                    sensor_max_range = 100.0)
    vanila_inverse.parallel_diagnostics()
