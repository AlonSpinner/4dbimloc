from typing import Tuple
from bim4loc.geometry.pose2z import compose_s, s_from_Rt
from bim4loc.geometry.scan_matcher.scan_matcher import scan_match
from bim4loc.existance_mapping.filters import exact, exact2
from bim4loc.random.multi_dim import sample_normal, SigmaPoints
from ..utils import low_variance_sampler
import numpy as np
import logging
from typing import Callable
from bim4loc.solids import IfcSolid
from bim4loc.sensors.sensors import Lidar
from bim4loc.maps import RayCastingMap
from bim4loc.random.multi_dim import gauss_likelihood

class RBPF():
    def __init__(self,
                simulation : RayCastingMap,
                sensor : Lidar,
                resample_rate : int,
                U_COV : np.ndarray):
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

        U_COV_remove_z = np.zeros((3,3))
        U_COV_remove_z[:2,:2] = U_COV[:2,:2]
        U_COV_remove_z[2,:] = U_COV[3,[0,1,3]]
        U_COV_remove_z[:,2] = U_COV[[0,1,3],3]
        sigmapoints = SigmaPoints(n = 3, alpha = 1.2, beta = 2.0, mu = np.zeros(3), cov = U_COV_remove_z)
        self._sigmapoints = sigmapoints

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
            pdf_scan_match = gauss_likelihood(s_from_Rt(R,t),np.zeros(4),U_COV)
            if rmse < 0.5 and pdf_scan_match > 0.5: #downsample_voxelsize = 0.5
                particle_poses[k] = compose_s(particle_poses[k], s_from_Rt(R,t))
                particle_z_values, particle_z_ids, _, \
                particle_z_cos_incident, particle_z_d = self._sense_fcn(particle_poses[k])
        
            #sample around the mode and average pz and particle beliefs

            #remap and calcualte probability of rays pz
            new_particle_beliefs = np.zeros_like(particle_beliefs[k])
            pz = np.zeros_like(z)

            for sigmapoint, sigmapoint_weight in zip(self._sigmapoints.points, self._sigmapoints.weights):
                particle_beliefs_m = particle_beliefs[k].copy()
                pose_m = compose_s(particle_poses[k], sigmapoint)

                #sense
                particle_z_values, particle_z_ids, _, \
                particle_z_cos_incident, particle_z_d = self._sense_fcn(pose_m)

                particle_beliefs_m, pz_m = exact2(pose_m,
                                                self._simulation_solids,
                                                particle_beliefs[k].copy(), 
                                                weights[k],
                                                z, 
                                                particle_z_values, 
                                                particle_z_ids, 
                                                particle_z_cos_incident,
                                                particle_z_d,
                                                self._sensor.uv,
                                                self._sensor.std,
                                                self._sensor.max_range)
                new_particle_beliefs = new_particle_beliefs + sigmapoint_weight * particle_beliefs_m
                pz = pz + sigmapoint_weight * pz_m 
            particle_beliefs[k] = new_particle_beliefs 
            
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

def generate_sigma_points(mu, cov, beta = 2, alpha = 1, n = 2):
    κ = 3 - n
    λ = alpha^2 * (n+κ) - n
    M = np.sqrt(n+λ)*cov
    sigma_points = np.zeros((2*n+1, n))
    sigma_points[0] = mu
    for i in range(n):
        sigma_points[i+1] = mu + M[i]
        sigma_points[i+1+n] = mu - M[i]


    L = np.linalg.cholesky((n + alpha) * np.cov(normal))
    sigma_points = np.zeros((2*n, normal.shape[1]))
    sigma_points[:n] = normal + L
    sigma_points[n:] = normal - L
    return sigma_points
