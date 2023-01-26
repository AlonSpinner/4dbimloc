from bim4loc.geometry.pose2z import compose_s, s_from_Rt
from bim4loc.geometry.scan_matcher.scan_matcher import scan_match
from bim4loc.existance_mapping.filters import exact_robust as existence_filter
from ..utils import low_variance_sampler
import numpy as np
from bim4loc.random.multi_dim import gauss_likelihood, sample_normal
from .bimloc import RBPF as RBPF_FULL

class RBPF(RBPF_FULL):
    def __init__(self,*args, **kwargs):
        super(RBPF, self).__init__(*args, **kwargs)

    def step(self, u, z):
        '''
        u - delta pose, array of shape (4)
        z - lidar scan, array of shape (N_lidar_beams)
        '''
        #initalize
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
        
            self.particle_beliefs[k], pz, self._particle_reservoirs[k] = existence_filter(self.particle_poses[k],
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
