import numpy as np
from bim4loc.geometry import Pose2z
from bim4loc.maps import Map
from bim4loc.agents import Drone
from bim4loc.random_models.multi_dim import gauss_likelihood, gauss_fit
import logging
import time
from copy import deepcopy

class vanila:
    def __init__(self, agent : Drone, m : Map ,initial_states : list[Pose2z]):
        self.agent : Drone = agent

        self.N_PARTICLES : int = len(initial_states) #amount of particles
        self.STATE_SIZE : int = 4
        
        self.particles = initial_states
        self.m = m # map must have method forward_measurement_model(x)
        
        self.weights : np.ndarray = np.ones(self.N_PARTICLES) * 1/self.N_PARTICLES
        self.ETA_THRESHOLD : float = 4.0/self.N_PARTICLES # bigger - lower threshold
        self.SPREAD_THRESHOLD = 1.0 #bigger - higher threshold

        self.init_time = time.time()

    def step(self, z : np.ndarray ,z_cov : np.ndarray,
                    u : Pose2z ,u_cov : np.ndarray):
        
        #update particles
        for i in range(self.N_PARTICLES):
            
            #create proposal distribution
            whiten_u = Pose2z(*np.random.multivariate_normal(u.Log(),u_cov))
            self.particles[i] = self.particles[i].compose(whiten_u)
            
            #create target distribution
            zhat = self.m.forward_measurement_model(self.particles[i], 
                                                    angles = self.agent.lidar_angles, 
                                                    max_range = self.agent.lidar_max_range)

            self.weights[i] *= gauss_likelihood(z, zhat, z_cov, pseudo = False)

        #normalize
        sm = self.weights.sum()
        if sm == 0.0: #numerical errors can cause this if particles have diverged from solution
            logging.warning(f'{time.time() - self.init_time:2.2f}[s]: numerically caused weight reset')
            self.weights = np.ones(self.N_PARTICLES) * 1/self.N_PARTICLES
        else:
            self.weights = self.weights/sm
            
        #resample
        n_eff = self.weights.dot(self.weights)
        if n_eff < self.ETA_THRESHOLD:
            logging.info(f'{time.time() - self.init_time}[s]: resampling')
            self.resample()

    def resample(self):
        #low_variance_sampler
        r = np.random.uniform()/self.N_PARTICLES
        idx = 0
        c = self.weights[idx]
        new_particles = []
        for i in range(self.N_PARTICLES):
            u = r + i*1/self.N_PARTICLES
            while u > c:
                idx += 1
                c += self.weights[idx]
            new_particles.append(deepcopy(self.particles[idx]))
        
        self.particles = new_particles
        self.weights = np.ones(self.N_PARTICLES) * 1/self.N_PARTICLES