import numpy as np
from bim4loc.geometry import pose2
from bim4loc.maps import Map
from bim4loc.agents import Drone
from bim4loc.random_models.multi_dim import gauss_likelihood, gauss_fit

import time

START_TIME = time.time()

class vanila_SE2:
    def __init__(self, agent : Drone, m : Map ,initial_states : list[pose2]):
        self.agent : Drone = agent

        self.N_PARTICLES : int = len(initial_states) #amount of particles
        self.STATE_SIZE : int = 3
        
        self.particles = initial_states
        self.m = m # map must have method forward_measurement_model(x)
        
        self.weights : np.ndarray = np.ones(self.N_PARTICLES) * 1/self.N_PARTICLES
        self.ETA_THRESHOLD : float = 4.0/self.N_PARTICLES # bigger - lower threshold
        self.SPREAD_THRESHOLD = 1.0 #bigger - higher threshold

        self.verbose = True

    def step(self, z : np.ndarray ,z_cov : np.ndarray,
                    u : pose2 ,u_cov : np.ndarray):
        
        #update particles
        for i in range(self.N_PARTICLES):
            
            #create proposal distribution
            whiten_u = pose2(*np.random.multivariate_normal(u.local(),u_cov))
            self.particles[i] = self.particles[i] + whiten_u
            
            #create target distribution
            zhat = self.m.forward_measurement_model(self.particles[i], angles = self.agent.lidar_angles)


            self.weights[i] *= gauss_likelihood(z, zhat, z_cov, pseudo = True)

        #normalize
        sm = self.weights.sum()
        if sm == 0.0: #numerical errors can cause this if particles have diverged from solution
            if self.verbose: print(f'{time.time() - START_TIME}[s]: numerically caused weight reset')
            self.weights = np.ones(self.N_PARTICLES) * 1/self.N_PARTICLES
        else:
            self.weights = self.weights/sm
            
        #resample
        spread = np.linalg.norm(np.cov(self.particleLocals().T))
        n_eff = self.weights.dot(self.weights)
        if n_eff < self.ETA_THRESHOLD or spread > self.SPREAD_THRESHOLD:
            if self.verbose: print(f'{time.time() - START_TIME}[s]: resampling')
            self.low_variance_sampler()

    def low_variance_sampler(self):
        r = np.random.uniform()/self.N_PARTICLES
        idx = 0
        c = self.weights[idx]
        new_particles = []
        for i in range(self.N_PARTICLES):
            u = r + i*1/self.N_PARTICLES
            while u > c:
                idx += 1
                c += self.weights[idx]
            new_particles.append(self.particles[idx])
        
        self.particles = new_particles
        self.weights = np.ones(self.N_PARTICLES) * 1/self.N_PARTICLES

    def estimateGaussian(self):
        locals = self.particleLocals().T # -> 3xN_PARTICLES
        mu, cov = gauss_fit(locals, self.weights)       
        return mu,cov

    def particleLocals(self):
        return np.array([p.local() for p in self.particles])