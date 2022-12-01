from typing import Tuple
from bim4loc.geometry.pose2z import compose_s, s_from_Rt
from bim4loc.geometry.scan_matcher.scan_matcher import scan_match
from bim4loc.existance_mapping.filters import exact, exact2
from bim4loc.random.multi_dim import sample_normal
from ..utils import low_variance_sampler
import numpy as np
import logging
from typing import Callable
from bim4loc.solids import IfcSolid
from bim4loc.sensors.sensors import Lidar
from bim4loc.maps import RayCastingMap

class RBPF():
    def __init__(self,
                simulation : RayCastingMap,
                sensor : Lidar,
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
        self._simulation_solids = simulation.solids
        self._sensor = sensor
        self._sense_fcn = lambda x: sensor.sense(x, simulation, n_hits = 5, noisy = False)
        self._scan_to_points_fcn = sensor.get_scan_to_points()

        map_bounds_min, map_bounds_max, extent = simulation.bounds()
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

            # if particle moved outside the map, kill it?
            if np.any(particle_poses[k][:3] < self._map_bounds_min[:3]) \
                or np.any(particle_poses[k][:3] > self._map_bounds_max[:3]):
                weights[k] = 0.0
                continue

            #sense
            particle_z_values, particle_z_ids, _, \
            particle_z_cos_incident, particle_z_d = self._sense_fcn(particle_poses[k])

            R,t, rmse = scan_match(z, particle_z_values, particle_z_ids,
                particle_beliefs[k],
                self._sensor.std, self._sensor.max_range,
                self._scan_to_points_fcn,
                self._scan_to_points_fcn,
                downsample_voxelsize = 0.5,
                icp_distance_threshold = 10.0,
                probability_filter_threshold = 0.3)
            if rmse < 0.5:
                particle_poses[k] = compose_s(particle_poses[k], s_from_Rt(R,t))
                particle_z_values, particle_z_ids, _, \
                particle_z_cos_incident, particle_z_d = self._sense_fcn(particle_poses[k])
        
            #remap and calcualte probability of rays pz
            particle_beliefs[k], pz = exact2(particle_poses[k],
                                            self._simulation_solids,
                                            particle_beliefs[k], 
                                            z, 
                                            particle_z_values, 
                                            particle_z_ids, 
                                            particle_z_cos_incident,
                                            particle_z_d,
                                            self._sensor.uv,
                                            self._sensor.std,
                                            self._sensor.max_range)
            
            # weights[k] *= np.product(pz) #or multiply?
            weights[k] *= 1.0 + np.sum(pz)
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