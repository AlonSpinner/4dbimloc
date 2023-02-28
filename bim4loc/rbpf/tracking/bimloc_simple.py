from bim4loc.geometry.pose2z import compose_s, s_from_Rt
from bim4loc.geometry.scan_matcher.scan_matcher import scan_match
from bim4loc.existance_mapping.filters import exact_simple as existence_filter
import numpy as np
from bim4loc.random.multi_dim import gauss_likelihood, sample_normal
from .bimloc_robust import RBPF as RBPF_FULL
import logging

class RBPF(RBPF_FULL):
    def __init__(self,*args, **kwargs):
        super(RBPF, self).__init__(*args, **kwargs)
        self._particle_reservoirs = None

    def step(self, u, z):
        '''
        u - delta pose, array of shape (4)
        z - lidar scan, array of shape (N_lidar_beams)
        '''
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
            _, _ = self._sense_fcn(self.particle_poses[k])

            registered, dpose = self.register2map(z, self.particle_beliefs[k], particle_z_values, particle_z_ids)
            if registered:
                self.particle_poses[k] = compose_s(self.particle_poses[k], dpose)
                particle_z_values, particle_z_ids, _, \
                _, _ = self._sense_fcn(self.particle_poses[k])
        
            self.particle_beliefs[k], pz = existence_filter(
                                            self.particle_beliefs[k].copy(), 
                                            z, 
                                            particle_z_values, 
                                            particle_z_ids, 
                                            self._sensor.std,
                                            self._sensor.max_range,
                                            self._sensor.p0)
            
            for variation in self._solids_varaition_dependence:
                for e_k in variation:
                    if self.particle_beliefs[k][e_k] > 0.9:
                        self.particle_beliefs[k][e_k] = 1.0
                        self.particle_beliefs[k][variation[variation != e_k]] = 0.0
                        break

            for key in self._solids_existence_dependence:
                #key's existence depends on value's existence
                value = self._solids_existence_dependence[key]
                self.particle_beliefs[k][value] = max(self.particle_beliefs[k][key],self.particle_beliefs[k][value])

            self.weights[k] = self.compute_weight(pz, z, self.weights[k])
            sum_weights += self.weights[k]

        #normalize weights
        if sum_weights < 1e-16: #prevent divide by zero
            self.weights = np.ones(self._N) / self._N
        else:
            self.weights /= sum_weights

        #resample
        if self.N_eff() < self._N/2 or self._step_counter % self._max_steps_to_resample == 0:
            self.resample()
        self._step_counter += 1
