from bim4loc.geometry.pose2z import compose_s, s_from_Rt
from bim4loc.geometry.scan_matcher.scan_matcher import scan_match
from bim4loc.existance_mapping.filters import exact2
from ..utils import low_variance_sampler
import numpy as np
from bim4loc.sensors.sensors import Lidar
from bim4loc.maps import RayCastingMap
from bim4loc.random.multi_dim import gauss_likelihood, sample_normal, gauss_fit

class RBPF():
    def __init__(self,
                simulation : RayCastingMap,
                sensor : Lidar,
                initial_particle_poses : np.ndarray,
                initial_belief : np.ndarray,
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

        self._U_COV = U_COV
        self._reservoir_decay_rate = 0.5

        self._N = initial_particle_poses.shape[0]
        self.particle_poses = initial_particle_poses
        self.particle_beliefs = np.tile(initial_belief, (self._N,1))
        self._particle_reservoirs = np.zeros((self._N, len(self._simulation_solids)))
        self.weights = np.ones(self._N) / self._N

    def N_eff(self):
        return 2.0 / np.sum(self.weights**2)

    def get_expected_belief_map(self):
        return np.sum(self.weights.reshape(-1,1) * self.particle_beliefs, axis = 0)

    def get_expect_pose(self):
        mu, cov = gauss_fit(self.particle_poses.T, self.weights)
        return mu, cov

    def decay_reservoirs(self):
        self._particle_reservoirs = self._particle_reservoirs * np.exp(-self._reservoir_decay_rate)

    def step(self, u, z):
        '''
        u - delta pose, array of shape (4)
        z - lidar scan, array of shape (N_lidar_beams)
        '''
        self.decay_reservoirs()

        #compute weights and normalize
        sum_weights = 0.0
        noisy_u_array = sample_normal(u, self._U_COV, self._N)
        for k in range(self._N):
            #move with scan matching
            self.particle_poses[k] = compose_s(self.particle_poses[k], noisy_u_array[k])

            # if particle moved outside the map, kill it
            if np.any(self.particle_poses[k][:3] < self._map_bounds_min[:3]) \
                or np.any(self.particle_poses[k][:3] > self._map_bounds_max[:3]):
                self.weights[k] = 0.0
                continue

            #sense
            particle_z_values, particle_z_ids, _, \
            particle_z_cos_incident, particle_z_d = self._sense_fcn(self.particle_poses[k])

            R,t, rmse = scan_match(z, particle_z_values, particle_z_ids,
                self.particle_beliefs[k],
                self._sensor.std, self._sensor.max_range,
                self._scan_to_points_fcn,
                self._scan_to_points_fcn,
                downsample_voxelsize = 0.5,
                icp_distance_threshold = 10.0,
                probability_filter_threshold = 0.3)
            pdf_scan_match = gauss_likelihood(s_from_Rt(R,t),np.zeros(4),self._U_COV)
            if rmse < 0.5 and pdf_scan_match > 0.05: #downsample_voxelsize = 0.5
                self.particle_poses[k] = compose_s(self.particle_poses[k], s_from_Rt(R,t))
                particle_z_values, particle_z_ids, _, \
                particle_z_cos_incident, particle_z_d = self._sense_fcn(self.particle_poses[k])
        
            self.particle_beliefs[k], pz, self._particle_reservoirs[k] = exact2(self.particle_poses[k],
                                            self._simulation_solids,
                                            self.particle_beliefs[k].copy(), 
                                            self.weights[k],
                                            self._particle_reservoirs[k].copy(),
                                            z, 
                                            particle_z_values, 
                                            particle_z_ids, 
                                            particle_z_cos_incident,
                                            particle_z_d,
                                            self._sensor.uv,
                                            self._sensor.std,
                                            self._sensor.max_range,
                                            self._sensor.p0)
            
            # weights[k] *= np.product(pz) #or multiply?
            self.weights[k] *= 1.0 + np.sum(pz)
            sum_weights += self.weights[k]

        #normalize weights
        if sum_weights < 1e-16: #prevent divide by zero
            self.weights = np.ones(self._N) / self._N
        else:
            self.weights /= sum_weights

        #resample
        if self.N_eff() < self._N:
            self.particle_poses, self.particle_beliefs = low_variance_sampler(self.weights, 
                                                                    self.particle_poses, 
                                                                    self.particle_beliefs, 
                                                                    self._N)
