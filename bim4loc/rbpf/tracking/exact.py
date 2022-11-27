from typing import Tuple
from bim4loc.geometry.pose2z import compose_s
from bim4loc.existance_mapping.filters import exact, exact2
from bim4loc.random.multi_dim import sample_normal
from ..utils import low_variance_sampler
import numpy as np
import logging
from typing import Callable

class RBPF():
    def __init__(self,
                sense_fcn : Callable,
                sensor_std : float,
                sensor_max_range : float,
                map_bounds_min : np.ndarray,
                map_bounds_max : np.ndarray,
                resample_rate : int):
        '''
        PARAMTERS
        sense_fcn - NUMBA JITED NO PYTHON FUNCTION 
                        that takes in a particle pose and returns simulated measurements
        sensor_std - standard deviation of sensor measurements
        sensor_max_range - maximum range of sensor
        map_bounds_min, map_bounds_max - arrays of shape (3)
        resample_steps_thresholds - array of two elements.
        '''
        
        self._sense_fcn = sense_fcn
        self._sensor_std = sensor_std
        self._sensor_max_range = sensor_max_range
        self._map_bounds_min = map_bounds_min
        self._map_bounds_max = map_bounds_max
        self._steps_from_resample = 0
        self._resample_rate = resample_rate

    def step(self, particle_poses, particle_beliefs, weights,
                   u, U_COV, z):
        '''
        particle_poses - array of shape (N_particles, 4)
        particle_beliefs - array of shape (N_particles, N_cells)
        weights - array of shape (N_particles)
        u - delta pose, array of shape (4)
        U_COV - covariance matrix of delta pose, array of shape (4,4)
        z - lidar scan, array of shape (N_lidar_beams)
        '''
        N_particles = particle_poses.shape[0]

        #compute weights and normalize
        sum_weights = 0.0
        noisy_u = sample_normal(u, U_COV, N_particles)
        for k in range(N_particles):
            #move with scan matching

            particle_poses[k] = compose_s(particle_poses[k], noisy_u[k])

            #particle_poses[k] = scan_match()

            # if particle moved outside the map, kill it?
            # if np.any(particle_poses[k][:3] < self._map_bounds_min[:3]) \
            #     or np.any(particle_poses[k][:3] > self._map_bounds_max[:3]):
            #     weights[k] = 0.0
            #     continue
        
            #sense
            particle_z_values, particle_z_ids, _, _, _ = self._sense_fcn(particle_poses[k])

            #remap and calcualte probability of rays pz
            particle_beliefs[k], pz = exact(particle_beliefs[k], 
                                            z, 
                                            particle_z_values, 
                                            particle_z_ids, 
                                            self._sensor_std,
                                            self._sensor_max_range)
            
            weights[k] *= np.product(pz) #or multiply?
            sum_weights += weights[k]

        #normalize weights
        if sum_weights < 1e-16: #prevent divide by zero
            weights = np.ones(N_particles) / N_particles
        else:
            weights = weights / sum_weights

        #resample
        if self._steps_from_resample == self._resample_rate-1:
            particle_poses, particle_beliefs = low_variance_sampler(weights, particle_poses, particle_beliefs, N_particles)
            self._steps_from_resample = 0
        else:
            self._steps_from_resample += 1
        return particle_poses, particle_beliefs, weights