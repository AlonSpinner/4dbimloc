from bim4loc.random.utils import p2logodds, logodds2p
from bim4loc.geometry.raycaster import NO_HIT
import bim4loc.random.one_dim as r_1d
import numpy as np
from numba import njit
from numba import prange

EPS = 1e-16

@njit(cache = True)
def binary_variable_update(current, update):
    #from page 30 in Robotic Mapping and Exporation (Occpuancy Probability Mapping)
    n_update = 1.0 - update
    n_current = 1.0 - current
    update = max(update,EPS)
    current = max(current,EPS)
    
    return np.reciprocal(1.0 + n_current/current * n_update/update)

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

def new_forward_ray(wz_i, sz_i, szid_i, beliefs, sensor_std, sensor_max_range):
    N_maxhits = sz_i.size
    valid_hits = 0
    for j in prange(N_maxhits):
        if szid_i[j] == NO_HIT:
            break
        valid_hits += 1

    a = np.zeros(valid_hits) #a[j] holds partial probability of hitting cell j-1
    b = np.zeros(valid_hits) #b[j] holds product of previous cells being empty
    for j in prange(0, valid_hits):        
        if j == 0:
            a[j] = 0.0 #forward(wz_i, sz_ij, sensor_std, pseudo = True) * belief_ij
            b[j] = 1.0 # - belief_ij
        else:
            sz_ijm1 = sz_i[j -1]
            belief_ijm1 = beliefs[szid_i[j - 1]]
            a[j] = a[j-1] + b[j-1] * forward(wz_i, sz_ijm1, sensor_std, pseudo = True) * belief_ijm1
            b[j] = b[j-1] * (1.0 - belief_ijm1)

    c = np.zeros(valid_hits) #c[j] holds the partial probabiltiy that cells beyond were hit even though j should be hit
    for j in prange(valid_hits - 1, -1, -1):
        if j == valid_hits - 1:
            #if all cells are empty (we assume c[j] is not. its confusing)
            c[j] = b[j] * forward(wz_i, sensor_max_range, sensor_std, pseudo = True)
        else:
            sz_ij = sz_i[j]
            belief_ij = beliefs[szid_i[j]]

            szid_ijp1 = sz_i[j+1]
            belief_ijp1 = beliefs[szid_i[j+1]]

            c[j] = belief_ijp1/(1.0 - belief_ij)*c[j+1] + \
                +b[j] * forward(wz_i, szid_ijp1, sensor_std, pseudo = True) * belief_ijp1

    pz_ij = np.zeros(valid_hits)
    npz_ij = np.zeros(valid_hits)
    for j in prange(valid_hits):
        sz_ij = sz_i[j]
        belief_ij = beliefs[szid_i[j]]

        pz_ij[j] = a[j] + b[j] * forward(wz_i, sz_ij, sensor_std, pseudo = True)
        npz_ij[j] = a[j] + c[j]

    return pz_ij, npz_ij

def new_vanila_forward(beliefs : np.ndarray, 
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
        pz_ij, npz_ij = new_forward_ray(wz_i, sz_i, szid_i, beliefs, sensor_std, sensor_max_range)

        for j in prange(N_maxhits):
            if szid_i[j] == NO_HIT:
                break
            szid_ij = szid_i[j]
            d = pz_ij[j] * beliefs[szid_ij]
            e = npz_ij[j] * (1.0 - beliefs[szid_ij])
            beliefs[szid_ij] = d / (d + e)
    
    return beliefs

# @njit(parallel = True, cache = True)
def forward_ray(wz_i, sz_i, szid_i, beliefs, sensor_std, sensor_max_range):
    '''
    inputs:
    world_z - single range measurement from real-world-sensor
    belief_z - np.ndarray
    sensor_std - both of real and simulated senosr (must be equivalent)
    pm - belief_m - probablity that solid m does exist
    '''
    N_maxhits = sz_i.size

    pz_ij = np.zeros_like(sz_i)
    pierced_meshes_probability = 1.0
    pdf_sum = 0.0
    for j in prange(N_maxhits):
        sz_ij = sz_i[j]
        szid_ij = szid_i[j]
        if szid_ij == NO_HIT: # hits are sorted from close->far->NO_HIT. so nothing to do anymore
            break

        belief_ij = beliefs[szid_ij]
        f_ij = forward(wz_i, sz_ij, sensor_std, pseudo = True)

        pz_ij[j] = pierced_meshes_probability * f_ij * belief_ij + temp
        temp = pz_ij

        #update pierced meshes probability for next iteration
        pierced_meshes_probability = pierced_meshes_probability * (1.0 - belief_ij)
    
    pz_max_range = pierced_meshes_probability * forward(wz_i, sensor_max_range, sensor_std, pseudo = True)
    pz_random = 1/sensor_max_range
    pz_i = np.sum(pz_ij) + pz_random + pz_max_range
    
    return pz_ij, pz_i

# @njit(parallel = True, cache = True)
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
        pz_ij, pz_i = forward_ray(wz_i, sz_i, szid_i, beliefs, sensor_std, sensor_max_range)

        for j in prange(N_maxhits):
            if szid_i[j] == NO_HIT:
                break
            szid_ij = szid_i[j]
            inv_pz_ij = pz_ij[j]/pz_i# * beliefs[szid_ij]/pz_i
            beliefs[szid_ij] = binary_variable_update(beliefs[szid_ij],inv_pz_ij)

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
