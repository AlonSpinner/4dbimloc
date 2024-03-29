from bim4loc.geometry.pose2z import compose_s, s_from_Rt
from bim4loc.geometry.scan_matcher.scan_matcher import scan_match
import numpy as np
from bim4loc.random.multi_dim import gauss_likelihood, sample_normal
from bim4loc.random.utils import logodds2p, p2logodds
from bim4loc.sensors.models import inverse_lidar_model
from bim4loc.existance_mapping.filters import approx_logodds as existence_filter
from .bimloc_robust import RBPF as RBPF_FULL
import logging

class RBPF(RBPF_FULL):
    def __init__(self,*args, **kwargs):
        super(RBPF, self).__init__(*args, **kwargs)
        self._logodds_inital_belief = p2logodds(args[3].copy())
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
        
            #calcualte importance weight -> find current posterior distribution
            pz = np.zeros(len(z))
            for j in range(len(z)):
                _, pz[j] = inverse_lidar_model(z[j], 
                                            particle_z_values[j],
                                            particle_z_ids[j], 
                                            self.particle_beliefs[k], 
                                self._sensor.std, self._sensor.max_range, self._sensor.p0)

            logodds_particle_beliefs = existence_filter(p2logodds(self.particle_beliefs[k].copy()), 
                                        z, 
                                        particle_z_values, 
                                        particle_z_ids, 
                                        self._sensor.std,
                                        self._sensor.max_range,
                                        self._logodds_inital_belief)
            self.particle_beliefs[k] = logodds2p(logodds_particle_beliefs)
            
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
