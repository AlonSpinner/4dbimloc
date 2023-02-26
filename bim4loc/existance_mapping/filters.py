from bim4loc.random.utils import p2logodds, p2odds, negate, logodds2p
from bim4loc.geometry.raycaster import NO_HIT
import bim4loc.random.one_dim as r_1d
import numpy as np
from numba import njit
from numba import prange
import matplotlib.pyplot as plt
from bim4loc.sensors.models import inverse_lidar_model
from bim4loc.geometry.pose2z import angle
from bim4loc.geometry.minimal_distance import minimal_distance_from_projected_boundry

EPS = 1e-16

#extract for numba performance
gaussian_pdf = r_1d.Gaussian._pdf 
exponentialT_pdf  = r_1d.ExponentialT._pdf

@njit(cache = True)
def binary_variable_update(current, update):
    #from page 30 in Robotic Mapping and Exporation (Occpuancy Probability Mapping
    return np.reciprocal(1.0 + p2odds(negate(current)) * p2odds(negate(update)))

# @njit(parallel = True, cache = True)
def exact_simple(beliefs : np.ndarray, 
            world_z : np.ndarray, 
            simulated_z : np.ndarray, 
            simulated_z_ids : np.ndarray,
            sensor_std : float,
            sensor_max_range : float,
            sensor_p0 : float) -> np.ndarray:
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
                                sensor_std, sensor_max_range, sensor_p0)
        for j, p in enumerate(pj_zi):
            szid_ij = szid_i[j] #simulated solid id of j'th hit in i'th ray
            beliefs[szid_ij] = p #binary_variable_update(beliefs[szid_ij], p)
        
        p_z[i] = p_z_i
        
    return beliefs, p_z

# @njit(parallel = True, cache = True)
def exact_robust(pose : np.ndarray,
            simulation_solids,
            beliefs : np.ndarray, 
            particle_weight : np.ndarray,
            particle_reservoir : np.ndarray,
            world_z : np.ndarray, 
            simulated_z : np.ndarray, 
            simulated_z_ids : np.ndarray,
            simulated_z_cos : np.ndarray,
            sensor_uv_angles : np.ndarray,
            sensor_std : float,
            sensor_max_range : float,
            sensor_p0 : float,
            semi_robust = False) -> np.ndarray:
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
    intersection_weights = np.zeros((N_elements,N_rays))

    for i in prange(N_rays):
        wz_i = world_z[i]
        sz_i = simulated_z[i]
        szid_i =  simulated_z_ids[i]

        pj_zi, p_z_i = inverse_lidar_model(wz_i, sz_i, szid_i, beliefs, 
                                sensor_std, sensor_max_range, sensor_p0)
        for j, p in enumerate(pj_zi):
            szid_ij = szid_i[j] #simulated solid id of j'th hit in i'th ray
            new_beliefs[szid_ij,i] = p
            intersection_weights[szid_ij,i] =  simulated_z_cos[i,j]**2
        
        p_z[i] = p_z_i
    
    for i in prange(N_elements):
        if beliefs[i] == 1.0 or beliefs[i] == 0.0: #won't be updated anyway
            continue
        element_new_beliefs = new_beliefs[i,:]
        indicies = element_new_beliefs >= 0

        if np.any(indicies) > 0:
            element_world_v = np.asarray(simulation_solids[i].geometry.vertices)
            element_uv = angle(np.ascontiguousarray(pose), element_world_v.T).T
            
            intersection_uv = sensor_uv_angles[indicies]
            
            
            element_intersection_beliefs = element_new_beliefs[indicies]
            
            dist_2_boundry = np.zeros(len(intersection_uv))
            for j, bearing in enumerate(intersection_uv):
                dist_2_boundry[j], _ = minimal_distance_from_projected_boundry(bearing, element_uv)
            element_intersection_weights = intersection_weights[i, indicies] * dist_2_boundry
            
            sum_element_weights = np.sum(element_intersection_weights) + EPS
            new_element_belief = np.sum(element_intersection_beliefs * element_intersection_weights/sum_element_weights)
            
            if semi_robust:
                beliefs[i] = new_element_belief
            else:
                a = particle_weight
                beliefs[i] = (new_element_belief * sum_element_weights + \
                            beliefs[i] * a*particle_reservoir[i]) / (sum_element_weights + a*particle_reservoir[i] + EPS)
                particle_reservoir[i] += sum_element_weights

    return beliefs, p_z, particle_reservoir

def approx_logodds_robust(pose : np.ndarray,
            simulation_solids,
            beliefs : np.ndarray, 
            particle_weight : np.ndarray,
            particle_reservoir : np.ndarray,
            world_z : np.ndarray, 
            simulated_z : np.ndarray, 
            simulated_z_ids : np.ndarray,
            simulated_z_cos : np.ndarray,
            sensor_uv_angles : np.ndarray,
            sensor_std : float,
            sensor_max_range : float,
            ) -> np.ndarray:
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
    N_maxhits = simulated_z.shape[1]

    new_beliefs = -np.ones((N_elements,N_rays))
    intersection_weights = np.zeros((N_elements,N_rays))

    T = 3 * sensor_std

    for i in prange(N_rays):
        wz_i = world_z[i]
        sz_i = simulated_z[i]
        szid_i =  simulated_z_ids[i]

        valid_hits = 0
        for j in prange(N_maxhits):
            if szid_i[j] == NO_HIT:
                break
            valid_hits += 1
        pj_zi = -np.ones(valid_hits)
            
        for j in prange(valid_hits):
            sz_ij = sz_i[j]
            szid_ij = szid_i[j]

            if szid_ij == NO_HIT or sz_ij >= sensor_max_range: # hits are sorted from close->far->NO_HIT. so nothing to do anymore
                break
            
            if wz_i < sz_ij - T: #wz had hit something before bzid
                break #same as adding 0.5
            elif wz_i < sz_ij + T: #wz has hit bzid
                pj_zi[j] = 0.5 + 0.2*gaussian_pdf(sz_ij, sensor_std, wz_i, pseudo = True)
            else: #wz has hit something after bzid, making us think bzid does not exist
                pj_zi[j] = 0.3

        for j, p in enumerate(pj_zi):
            szid_ij = szid_i[j] #simulated solid id of j'th hit in i'th ray
            new_beliefs[szid_ij,i] = p
            intersection_weights[szid_ij,i] =  simulated_z_cos[i,j]**2
        
    for i in prange(N_elements):
        if beliefs[i] == 1.0 or beliefs[i] == 0.0: #won't be updated anyway
            continue
        element_new_beliefs = new_beliefs[i,:]
        indicies = element_new_beliefs >= 0

        if np.any(indicies) > 0:
            element_world_v = np.asarray(simulation_solids[i].geometry.vertices)
            element_uv = angle(np.ascontiguousarray(pose), element_world_v.T).T
            
            nrm_intersection_uv = sensor_uv_angles[indicies]/np.array([np.pi, np.pi/2])
            nrm_element_uv = element_uv/np.array([np.pi, np.pi/2])
            
            
            element_intersection_beliefs = element_new_beliefs[indicies]
            
            dist_2_boundry = np.zeros(len(nrm_intersection_uv))
            for j, bearing in enumerate(nrm_intersection_uv):
                dist_2_boundry[j], _ = minimal_distance_from_projected_boundry(bearing, nrm_element_uv)
            element_intersection_weights = intersection_weights[i, indicies] * dist_2_boundry
            
            sum_element_weights = np.sum(element_intersection_weights) + EPS
            new_element_belief = np.sum(element_intersection_beliefs * element_intersection_weights/sum_element_weights)
            
            beliefs[i] = binary_variable_update(beliefs[i],new_element_belief)
            # a = particle_weight
            # beliefs[i] = logodds2p((p2logodds(new_element_belief) * sum_element_weights + \
            #             p2logodds(beliefs[i]) * a*particle_reservoir[i]) / \
            #                 (sum_element_weights + a*particle_reservoir[i] + EPS))
            # particle_reservoir[i] += sum_element_weights

    return beliefs

@njit(parallel = True, cache = True)
def approx_logodds(logodds_beliefs : np.ndarray, 
            world_z : np.ndarray, 
            simulated_z : np.ndarray, 
            simulated_z_ids : np.ndarray,
            sensor_std : float,
            sensor_max_range : float,
            logodds_inital_belief : np.ndarray) -> np.ndarray:
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

    for i in prange(N_rays):
        wz_i = world_z[i]
        sz_i = simulated_z[i]
        szid_i =  simulated_z_ids[i]
        
        for j in prange(N_maxhits):
            sz_ij = sz_i[j]
            szid_ij = szid_i[j]
            if szid_ij == NO_HIT or sz_ij >= sensor_max_range: # hits are sorted from close->far->NO_HIT. so nothing to do anymore
                break

            # if wz_i < sz_ij - T: #wz had hit something before bzid
            #     break #same as adding p2logodds(0.5)
            # elif wz_i < sz_ij + T: #wz has hit bzid
            #     logodds_beliefs[szid_ij] += p2logodds(0.5 + 0.3*gaussian_pdf(sz_ij, sensor_std, wz_i, pseudo = True))
            # else: #wz has hit something after bzid, making us think bzid does not exist
            #     logodds_beliefs[szid_ij] += p2logodds(0.2)

            if wz_i <= sz_ij:
                logodds_beliefs[szid_ij] += p2logodds(0.5 + 0.4*gaussian_pdf(sz_ij, sensor_std, wz_i, pseudo = True))
            else:
                logodds_beliefs[szid_ij] += p2logodds(0.4 + 0.5*gaussian_pdf(sz_ij, sensor_std, wz_i, pseudo = True))

    return logodds_beliefs

if __name__ == '__main__':
    approx_logodds(logodds_beliefs = np.array([0.5]),
                    world_z = np.array([0.5]), 
                    simulated_z = np.array([[0.5]]),
                    simulated_z_ids = np.array([[0]]),
                    sensor_std = 0.2,
                    sensor_max_range = 100.0)
    approx_logodds.parallel_diagnostics()
