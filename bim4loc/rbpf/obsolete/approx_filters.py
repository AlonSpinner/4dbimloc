from typing import Tuple
from numba import njit, prange
from bim4loc.geometry.pose2z import compose_s
from bim4loc.existance_mapping.filters import approx
from bim4loc.random.multi_dim import sample_normal
from .lpf_adaptive_resampling import resampler, update_resampling_ws
from bim4loc.sensors.models import inverse_lidar_model
from bim4loc.random.utils import logodds2p, p2logodds
from ..utils import should_resample
import numpy as np
import logging

def fast_slam_lpf_resampler(particle_poses, particle_beliefs, weights, u, U_COV, z, 
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

        #calcualte importance weight -> find current posterior distribution
        pz = np.zeros(len(z))
        for j in range(len(z)):
            _, pz[j] = inverse_lidar_model(z[j], 
                                           particle_z_values[j],
                                           particle_z_ids[j], 
                                           particle_beliefs[k], 
                            lidar_std, lidar_max_range)

        #remap and calcualte probability of rays pz
        logodds_particle_beliefs = approx(p2logodds(particle_beliefs[k]), 
                                        z, 
                                        particle_z_values, 
                                        particle_z_ids, 
                                        lidar_std,
                                        lidar_max_range)
        particle_beliefs[k] = logodds2p(logodds_particle_beliefs)
            
        weights[k] *= np.product(pz)
        sum_weights += weights[k]

    #----finished per particle calculations.

    w_slow, w_fast = update_resampling_ws(w_slow, w_fast, sum_weights, N_particles)

    #normalize weights
    if sum_weights == 0.0:
        weights = np.ones(N_particles) / N_particles
    weights = weights / sum_weights

    if should_resample(weights, steps_from_resample, resample_steps_thresholds):
        
        particle_poses, particle_beliefs, weights, w_slow, w_fast, w_diff = \
                 resampler(particle_poses, particle_beliefs, weights,
                            w_slow, w_fast,
                            pose_min_bounds, pose_max_bounds, 
                            map_initial_belief)
        steps_from_resample = 0
    else:
        steps_from_resample += 1
        w_diff = np.clip(1.0 - w_fast / w_slow, 0.0, 1.0)

    return particle_poses, \
        particle_beliefs, \
        weights, w_slow, w_fast, w_diff, steps_from_resample

            

    
