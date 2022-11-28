from bim4loc.random.utils import p2logodds, p2odds, negate
from bim4loc.geometry.raycaster import NO_HIT
import bim4loc.random.one_dim as r_1d
import numpy as np
from numba import njit
from numba import prange

from bim4loc.sensors.models import inverse_lidar_model

EPS = 1e-16

#extract for numba performance
gaussian_pdf = r_1d.Gaussian._pdf 
exponentialT_pdf  = r_1d.ExponentialT._pdf

@njit(cache = True)
def binary_variable_update(current, update):
    #from page 30 in Robotic Mapping and Exporation (Occpuancy Probability Mapping
    return np.reciprocal(1.0 + p2odds(negate(current)) * p2odds(negate(update)))

# @njit(parallel = True, cache = True)
def exact(beliefs : np.ndarray, 
            world_z : np.ndarray, 
            simulated_z : np.ndarray, 
            simulated_z_ids : np.ndarray,
            sensor_std : float,
            sensor_max_range : float) -> np.ndarray:
    '''
    CAREFUL. THIS FUNCTION ALTERS THE INPUT beliefs

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
        p_z - np.array of probability of measurements given existance beliefs
    '''

    N_rays = world_z.shape[0]
    p_z = np.zeros(N_rays)

    for i in prange(N_rays):
        wz_i = world_z[i]
        sz_i = simulated_z[i]
        szid_i =  simulated_z_ids[i]

        pj_zi, p_z_i = inverse_lidar_model(wz_i, sz_i, szid_i, beliefs, 
                                sensor_std, sensor_max_range)
        for j, p in enumerate(pj_zi):
            szid_ij = szid_i[j] #simulated solid id of j'th hit in i'th ray
            beliefs[szid_ij] = p #binary_variable_update(beliefs[szid_ij], p)
        
        p_z[i] = p_z_i
        
    return beliefs, p_z

# @njit(parallel = True, cache = True)
def exact2(beliefs : np.ndarray, 
            world_z : np.ndarray, 
            simulated_z : np.ndarray, 
            simulated_z_ids : np.ndarray,
            sensor_std : float,
            sensor_max_range : float) -> np.ndarray:
    '''
    CAREFUL. THIS FUNCTION ALTERS THE INPUT beliefs

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
        p_z - np.array of probability of measurements given existance beliefs
    '''

    N_elements = beliefs.shape[0]
    N_rays = world_z.shape[0]
    p_z = np.zeros(N_rays)

    new_beliefs = -np.ones((N_elements,N_rays))

    for i in prange(N_rays):
        wz_i = world_z[i]
        sz_i = simulated_z[i]
        szid_i =  simulated_z_ids[i]

        pj_zi, p_z_i = inverse_lidar_model(wz_i, sz_i, szid_i, beliefs, 
                                sensor_std, sensor_max_range)
        for j, p in enumerate(pj_zi):
            szid_ij = szid_i[j] #simulated solid id of j'th hit in i'th ray
            new_beliefs[szid_ij,i] = p
        
        p_z[i] = p_z_i
    
    for i in prange(N_elements):
        element_new_beliefs = new_beliefs[i,:]
        element_new_beliefs = element_new_beliefs[element_new_beliefs >= 0] #remove -1s

        hist, bin_edges = np.histogram(element_new_beliefs, bins = 20, range = (0,1))
        #create gaussian from previous belief with std ~ 
        # beliefs[i] = bin_edges[np.argmax(hist)]
       
        if element_new_beliefs.shape[0] > 0:
              beliefs[i] = np.max(element_new_beliefs)
        #     exist_beliefs = element_new_beliefs[element_new_beliefs > 0.5]
        #     if exist_beliefs.shape[0] > 0:
        #         beliefs[i] = np.mean(exist_beliefs)
            # else:
                # beliefs[i] = np.mean(element_new_beliefs)

    return beliefs, p_z

@njit(parallel = True, cache = True)
def approx(logodds_beliefs : np.ndarray, 
            world_z : np.ndarray, 
            simulated_z : np.ndarray, 
            simulated_z_ids : np.ndarray,
            sensor_std : float,
            sensor_max_range : float) -> np.ndarray:
    '''
    CAREFUL. THIS FUNCTION ALTERS THE INPUT logodds_beliefs

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
                # if not(logodds_beliefs[szid_ij] > 3.0): #<-----dont allow things to decrease from a certain range
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
