from numba import njit, prange
import numpy as np
import bim4loc.random.one_dim as r_1d
from bim4loc.random.utils import negate
gaussian_pdf = r_1d.Gaussian._pdf
gaussianT_pdf = r_1d.GaussianT._pdf
exponentialT_pdf  = r_1d.ExponentialT._pdf
from bim4loc.geometry.raycaster import NO_HIT
EPS = 1e-16


@njit(cache = True)
def forward_lidar_model(wz : np.ndarray, #wrapper for Gaussian_pdf
                        sz : np.ndarray, 
                        std : float, 
                        sensor_max_range) -> np.ndarray:
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
    # return gaussianT_pdf(mu = sz, sigma = std, x =  wz, a = 0, b = sensor_max_range)
    return gaussian_pdf(mu = sz, sigma = std, x =  wz, pseudo = False)

@njit(cache = True)
def delta(x,xmid):
    if x==xmid:
        return 1.0
    else:
        return 0.0

@njit(cache = True)
def inverse_lidar_model(wz_i, sz_i, szid_i, beliefs, 
                        sensor_std, sensor_max_range, sensor_p0 = 0.4):
    '''
    based on the awesome papers "Autonomous Exploration with Exact Inverse Sensor Models"
    and "Bayesian Occpuancy Grid Mapping via an Exact Inverse Sensor Model"
    by Evan Kaufman et al.
    
    input:
    wz_i - world z value of i'th ray (np. array of floats)
    sz_i - simulated z value of i'th ray (np.array of floats)
    beliefs - probability of existance of each solid (np.array of floats [0,1])
    sensor_std - standard deviation of sensor (float)
    sensor_max_range - maximum range of sensor (float)
    sensor_p0 - probablity density at range = 0

    output:
    pj_z_i - updated existance beliefs given measurement (probabilty of solid j, given z_i)
    p_z_i - probability of measurement given existance beliefs
    '''
    N_maxhits = sz_i.size
    valid_hits = 0
    for j in prange(N_maxhits):
        if szid_i[j] == NO_HIT:
            break
        valid_hits += 1

    #insert max range element, thank god creates new np arrays
    max_range_index = np.searchsorted(sz_i, sensor_max_range)
    sz_i = np.hstack((sz_i[:max_range_index], np.array([sensor_max_range]), sz_i[max_range_index:]))
    szid_i = np.hstack((szid_i[:max_range_index], np.array([len(beliefs)]), szid_i[max_range_index:]))
    valid_hits +=1

    Pjbar = 1.0
    inv_eta = 0.0
    pj_z_i_wave = np.zeros(valid_hits)

    #random hit
    p_random = exponentialT_pdf(sensor_p0, sensor_max_range, wz_i) #<<<--- super important to relax exact
    inv_eta += p_random
    inv_eta_normalizer = 1.0
    #solids
    Pjplus = 1.0
    for j in prange(valid_hits):
        sz_ij = sz_i[j]
        # sz_ij = min(sz_ij, sensor_max_range)

        if szid_i[j] == len(beliefs):
            belief_ij = (wz_i >= sensor_max_range)
            Pjplus = Pjbar * belief_ij
            Pjbar = Pjbar * negate(1.0)
            inv_eta +=  Pjplus * forward_lidar_model(sensor_max_range, sensor_max_range,sensor_std,sensor_max_range) * \
                (wz_i >= sensor_max_range)
            inv_eta_normalizer += Pjplus
            continue

        belief_ij = beliefs[szid_i[j]]

        Pjplus = Pjbar * belief_ij
        Pjbar = Pjbar * negate(belief_ij)
        
        a_temp = Pjplus * forward_lidar_model(wz_i, sz_ij,sensor_std, sensor_max_range)
        pj_z_i_wave[j] = (belief_ij * inv_eta + a_temp)
        inv_eta = inv_eta + a_temp

        inv_eta_normalizer += Pjplus
    
    pj_z_i_wave = np.hstack((pj_z_i_wave[:max_range_index],pj_z_i_wave[max_range_index+1:]))

    pj_z_i = pj_z_i_wave /inv_eta#max(inv_eta, EPS)
    p_z_i = inv_eta/inv_eta_normalizer
    return pj_z_i, p_z_i

# @njit(cache = True)
def inverse_lidar_model_PAPER_VERSION(wz_i, sz_i, szid_i, beliefs, 
                        sensor_std, sensor_max_range, sensor_p0 = 0.4):
    '''
    based on the awesome papers "Autonomous Exploration with Exact Inverse Sensor Models"
    and "Bayesian Occpuancy Grid Mapping via an Exact Inverse Sensor Model"
    by Evan Kaufman et al.
    
    input:
    wz_i - world z value of i'th ray (np. array of floats)
    sz_i - simulated z value of i'th ray (np.array of floats)
    beliefs - probability of existance of each solid (np.array of floats [0,1])
    sensor_std - standard deviation of sensor (float)
    sensor_max_range - maximum range of sensor (float)
    sensor_p0 - probablity density at range = 0

    output:
    pj_z_i - updated existance beliefs given measurement (probabilty of solid j, given z_i)
    p_z_i - probability of measurement given existance beliefs
    '''
    N_maxhits = sz_i.size
    valid_hits = 0
    for j in prange(N_maxhits):
        if szid_i[j] == NO_HIT:
            break
        valid_hits += 1

    #insert max range element, thank god creates new np arrays
    max_range_index = np.searchsorted(sz_i, sensor_max_range)
    sz_i = np.insert(sz_i, max_range_index, sensor_max_range)
    szid_i = np.insert(szid_i, max_range_index, len(beliefs))
    beliefs = np.insert(beliefs, len(beliefs), 1.0)
    valid_hits +=1

    Pjbar = 1.0
    inv_eta = 0.0
    pj_z_i_wave = np.zeros(valid_hits)

    #random hit
    p_random = exponentialT_pdf(sensor_p0, sensor_max_range, wz_i) #<<<--- super important to relax exact
    inv_eta += p_random
    inv_eta_normalizer = 1.0
    #solids
    for j in prange(valid_hits):
        sz_ij = sz_i[j]
        # sz_ij = min(sz_ij, sensor_max_range)

        belief_ij = beliefs[szid_i[j]]

        Pjplus = Pjbar * belief_ij
        Pjbar = Pjbar * negate(belief_ij)
        
        if szid_i[j] == len(beliefs)-1: #max range hit
            # a_temp = Pjplus * delta(wz_i, sensor_max_range)
            a_temp = Pjplus * forward_lidar_model(wz_i, sz_ij,sensor_std/100, sensor_max_range)
        else:
            a_temp = Pjplus * forward_lidar_model(wz_i, sz_ij,sensor_std, sensor_max_range)
        pj_z_i_wave[j] = (belief_ij * inv_eta + a_temp)
        inv_eta = inv_eta + a_temp

        inv_eta_normalizer += Pjplus
    
    pj_z_i = pj_z_i_wave/inv_eta
    p_z_i = inv_eta/inv_eta_normalizer

    pj_z_i = np.delete(pj_z_i, max_range_index)
    return pj_z_i, p_z_i

@njit(cache = True)
def inverse_lidar_model_old(wz_i, sz_i, szid_i, beliefs, 
                        sensor_std, sensor_max_range, sensor_p0 = 0.4):
    '''
    based on the awesome papers "Autonomous Exploration with Exact Inverse Sensor Models"
    and "Bayesian Occpuancy Grid Mapping via an Exact Inverse Sensor Model"
    by Evan Kaufman et al.
    
    input:
    wz_i - world z value of i'th ray (np. array of floats)
    sz_i - simulated z value of i'th ray (np.array of floats)
    beliefs - probability of existance of each solid (np.array of floats [0,1])
    sensor_std - standard deviation of sensor (float)
    sensor_max_range - maximum range of sensor (float)
    sensor_p0 - probablity density at range = 0

    output:
    pj_z_i - updated existance beliefs given measurement (probabilty of solid j, given z_i)
    p_z_i - probability of measurement given existance beliefs
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
    p_random = exponentialT_pdf(sensor_p0, sensor_max_range, wz_i) #<<<--- super important to relax exact
    inv_eta += p_random

    #solids
    for j in prange(valid_hits):
        sz_ij = sz_i[j]
        sz_ij = min(sz_ij, sensor_max_range)
        belief_ij = beliefs[szid_i[j]]

        Pjplus = Pjbar * belief_ij
        Pjbar = Pjbar * negate(belief_ij)
        
        a_temp = Pjplus * forward_lidar_model(wz_i, sz_ij, sensor_std, sensor_max_range)
        pj_z_i_wave[j] = belief_ij * inv_eta + a_temp
        inv_eta = inv_eta + a_temp
    
    #max range hit
    inv_eta += Pjbar * forward_lidar_model(wz_i, sensor_max_range, sensor_std,sensor_max_range)
    
    pj_z_i = pj_z_i_wave / max(inv_eta, EPS)
    p_z_i = inv_eta #/ (1.0 + p_random) #normalize so maximal value is 1. Dont want to do this.
    return pj_z_i, p_z_i