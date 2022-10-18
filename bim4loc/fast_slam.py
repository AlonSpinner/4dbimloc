from typing import Tuple
from numba import njit, prange
from bim4loc.geometry.pose2z import compose_s
from bim4loc.sensors.models import inverse_lidar_model
from bim4loc.random.utils import p2logodds,logodds2p
from bim4loc.existance_mapping.filters import approx, exact
from bim4loc.random.multi_dim import sample_uniform, sample_normal
import numpy as np
import logging

@njit(cache = True)
def low_variance_sampler(weights : np.ndarray, 
                         particle_poses : np.ndarray,
                         particle_beliefs : np.ndarray, 
                         N_resample : int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    input:
        weights - array of shape (N_particles)
        particle_poses - array of shape (N_particles, 4)
        particle_beliefs - array of shape (N_particles, N_cells)
        N_resample - number of particles to resample from N_particles

    output:
        new_particle_poses - array of shape (N_resample, 4)
        new_particle_beliefs - array of shape (N_resample, N_cells)

    slightly edited from Probalistic Robotics, Nonparametric filters, page 110
    '''
    new_particle_poses = np.zeros((N_resample,4))
    new_particle_beliefs = np.zeros((N_resample, particle_beliefs.shape[1]))
    
    idx = 0
    c = weights[0]
    duu = 1.0/N_resample
    r = np.random.uniform(0.0, 1.0) * duu
    
    for i in prange(N_resample):
        uu = r + i*duu
        while uu > c:
            idx += 1
            c += weights[idx]
        new_particle_poses[i] = particle_poses[idx]
        new_particle_beliefs[i] = particle_beliefs[idx]
    
    return new_particle_poses, new_particle_beliefs

@njit(cache = True, fastmath =True)
def update_resampling_ws(w_slow : float, w_fast : float,
                         sum_weights : float, N_particles : int,
                         alpha_slow = 0.001, alpha_fast = 2.0,) \
                         -> Tuple[float, float]:
    '''
    here we have two low pass filters, one fast and one slow.
    we later test the ratio between the two to see 
    if there was a quick change in the average weights for the worst.
    if there was, we will throw random particles from thin air into the mix

    input:
        w_slow, w_fast - filtered variables
        sum_weights - sum of weights of all particles before resampling
        N_particles - number of particles
        alpha_slow, alpha_fast - low pass filter constants

    output:
        new w_slow and w_fast
    '''
    w_avg = sum_weights / N_particles
    if w_slow == 0.0: #happens after filter resampling
        w_slow = w_avg
    else:
        w_slow = w_slow + alpha_slow * (w_avg - w_slow)
    if w_fast == 0.0: #happens after filter resampling
        w_fast = w_avg
    else:        
        w_fast = w_fast + alpha_fast * (w_avg - w_fast)
    
    return w_slow, w_fast

@njit(cache = True)
def should_resample(weights : np.ndarray, 
                    steps_from_resample : int, 
                    resample_steps_thresholds : np.ndarray) -> bool:
    '''
    input:
        weights - array of shape (N_particles)
        steps_from_resample - amount of particle filter steps since last resample
        resample_steps_thresholds - array of two elements.
            first element: minimum steps before resample can occur
            second element: amount of steps after which resample must occur

    output:
        True if we should resample, False otherwise
    '''
    N_particles = weights.shape[0]
    n_eff = weights.dot(weights)
    eta_threshold = 2.0/N_particles
    if n_eff < eta_threshold and steps_from_resample > resample_steps_thresholds[0] \
        or steps_from_resample >= resample_steps_thresholds[1]:
         return True
    else:
        return False

@njit(cache = True)
def filter_resampler(particle_poses : np.ndarray,
                     particle_beliefs : np.ndarray,
                     weights : np.ndarray,
                     w_slow : float, w_fast : float,
                     pose_min_bounds : np.ndarray, pose_max_bounds : np.ndarray, 
                     initial_belief : np.ndarray):
    '''
    called in the case resampling is needed

    input: 
        particle_poses - array of shape (N_particles, 4)
        particle_beliefs - array of shape (N_particles, N_cells)
        weights - array of shape (N_particles)
        w_slow, w_fast - floats for adaptive resampling
        pose_min_bounds, pose_max_bounds - arrays of shape (4)
        initial_belief - array of shape (N_cells)
    
    output:
        particle_poses - updated, 
        particle_beliefs - updated
        weights - updated, 
        w_slow, w_fast - updated, 
        w_diff - 0.0 if we have good samples, 1.0 if samples are trash
    '''
    N_particles = particle_poses.shape[0]
    
    #percentage of random samples
    w_diff = 1.0 - w_fast / w_slow
    if w_diff > 1.0: #clip it
        w_diff = 1.0
    elif w_diff < 0.0:
        w_diff = 0.0
    N_random = int(w_diff * N_particles)
    N_resample = N_particles - N_random
    
    N_resample = N_particles #for now, we will resample all particles
    N_random = 0

    #produce new samples from static distribuion with initial belief maps
    if N_random > 0:
        thinAir_particle_poses = sample_uniform(pose_min_bounds, pose_max_bounds, N_random)
        thinAir_particle_beliefs = initial_belief.repeat(N_random).reshape((-1, N_random)).T

    #produce resampled samples                                             
    if N_resample > 0:
        resampled_particle_poses, resampled_particle_beliefs = \
            low_variance_sampler(weights, particle_poses, particle_beliefs, N_resample)
    
    #mash them together
    if N_random == 0:
        particle_poses = resampled_particle_poses
        particle_beliefs = resampled_particle_beliefs
    elif N_resample == 0:
        particle_poses = thinAir_particle_poses
        particle_beliefs = thinAir_particle_beliefs
    else:
        particle_poses = np.vstack((thinAir_particle_poses, resampled_particle_poses))
        particle_beliefs = np.vstack((thinAir_particle_beliefs, resampled_particle_beliefs))

    #reset weights
    weights = np.ones(N_particles) / N_particles
    #reset averages, to avoid spiraling off into complete randomness.
    if w_diff > 0.0:
        w_slow = w_fast = 0.0
    
    return particle_poses, particle_beliefs, weights, w_slow, w_fast, w_diff

@njit(parallel = True, cache = True)
def fast_slam_filter(particle_poses, particle_beliefs, weights, u, U_COV, z, 
                    steps_from_resample, w_slow, w_fast,
                    sense_fcn, lidar_std, lidar_max_range, 
                    map_bounds_min, map_bounds_max, map_initial_belief,
                    resample_steps_thresholds = [4,6]):

    '''
    called in the case resampling is needed

    input:
        VARIABLES
        particle_poses - array of shape (N_particles, 4)
        particle_beliefs - array of shape (N_particles, N_cells)
        weights - array of shape (N_particles)
        u - delta pose, array of shape (4)
        U_COV - covariance matrix of delta pose, array of shape (4,4)
        z - lidar scan, array of shape (N_lidar_beams)
        steps_from_resample - amount of particle filter steps since last resample
        w_slow, w_fast - floats for adaptive resampling (are states in this filter)

        PARAMTERS
        sense_fcn - NUMBA JITED NO PYTHON FUNCTION 
                        that takes in a particle pose and returns simulated measurements
        lidar_std - standard deviation of lidar measurements
        lidar_max_range - maximum range of lidar
        map_bounds_min, map_bounds_max - arrays of shape (3)
        map_initial_belief - array of shape (N_cells)
        resample_steps_thresholds - array of two elements.

    output:
        particle_poses - updated, 
        particle_beliefs - updated
        weights - updated, 
        w_slow, w_fast - updated, 
        w_diff - w_fast/w_slow
        steps_from_resample - updated
    '''

    N_particles = particle_poses.shape[0]
    pose_min_bounds = np.array([map_bounds_min[0],map_bounds_min[1], 0.0 , -np.pi])
    pose_max_bounds = np.array([map_bounds_max[0],map_bounds_max[1], 0.0 , np.pi])

    #compute weights and normalize
    sum_weights = 0.0
    noisy_u = sample_normal(u, U_COV, N_particles)
    for k in prange(N_particles):
        #move
        particle_poses[k] = compose_s(particle_poses[k], noisy_u[k])

        #if particle moved outside the map, kill it?
        # if np.any(particle_poses[k][:3] < map_bounds_min[:3]) \
        #      or np.any(particle_poses[k][:3] > map_bounds_max[:3]):
        #     weights[k] = 0.0
        #     continue

        #sense
        particle_z_values, particle_z_ids, _, _, _ = sense_fcn(particle_poses[k])

        #remap and calcualte probability of rays pz
        particle_beliefs[k], pz = exact(particle_beliefs[k], 
                                        z, 
                                        particle_z_values, 
                                        particle_z_ids, 
                                        lidar_std,
                                        lidar_max_range)
            
        weights[k] *= pz.prod()
        sum_weights += weights[k]

    #----finished per particle calculations.

    w_slow, w_fast = update_resampling_ws(w_slow, w_fast, sum_weights, N_particles)

    #normalize weights
    if sum_weights == 0.0:
        weights = np.ones(N_particles) / N_particles
    weights = weights / sum_weights

    if should_resample(weights, steps_from_resample, resample_steps_thresholds):
        particle_poses, particle_beliefs, weights, w_slow, w_fast, w_diff = \
            filter_resampler(particle_poses, particle_beliefs, weights,
                            w_slow, w_fast,
                            pose_min_bounds, pose_max_bounds, 
                            map_initial_belief)
        steps_from_resample = 0
    else:
        steps_from_resample += 1

    return particle_poses, \
        particle_beliefs, \
        weights, w_slow, w_fast, w_diff, steps_from_resample

            

    
