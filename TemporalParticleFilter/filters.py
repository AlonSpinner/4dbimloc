import numpy as np
from particleFilter.geometry import pose2
from particleFilter.maps import Map
from particleFilter.gaussians import gauss_likelihood, gauss_fit
import time

START_TIME = time.time()

class pf_vanila_SE2:
    def __init__(self,m : Map ,initial_states : list[pose2]):
        self.N_PARTICLES : int = len(initial_states) #amount of particles
        self.STATE_SIZE : int = 3
        
        self.particles = initial_states
        self.m = m # map must have method forward_measurement_model(x)
        
        self.weights : np.ndarray = np.ones(self.N_PARTICLES) * 1/self.N_PARTICLES
        self.ETA_THRESHOLD : float = 4.0/self.N_PARTICLES # bigger - lower threshold
        self.SPREAD_THRESHOLD = 1.0 #bigger - higher threshold

        self.verbose = True
        return

    def step(self,z,z_cov,
                    u,u_cov):
        
        #update particles
        for i in range(self.N_PARTICLES):
            
            #create proposal distribution
            noise = np.random.multivariate_normal(np.zeros((self.STATE_SIZE)),u_cov)
            noise = pose2(noise[0],noise[1],noise[2])
            self.particles[i] = self.particles[i] + (u + noise)
            
            #create target distribution
            zhat = self.m.forward_measurement_model(self.particles[i])
            self.weights[i] *= gauss_likelihood(z,zhat,z_cov, pseudo = True)

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