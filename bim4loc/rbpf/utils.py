from numba import njit, prange
import numpy as np
from typing import Tuple

@njit(cache = True)
def low_variance_sampler(weights : np.ndarray, 
                         particle_poses : np.ndarray,
                         particle_beliefs : np.ndarray, 
                         N_resample : int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    input:
        weights - array of shape (N_particles)
        particle_poses - array of shape (N_particles, 4)
        particle_beliefs - array of shape (N_particles, N_cells)
        N_resample - number of particles to resample from N_particles

    output:
        new_particle_poses - array of shape (N_resample, 4)
        new_particle_beliefs - array of shape (N_resample, N_cells)

    slightly edited from Probalistic Robotics, Nonparametric filters, page 110
    '''
    new_particle_poses = np.zeros((N_resample,4))
    new_particle_beliefs = np.zeros((N_resample, particle_beliefs.shape[1]))
    
    idx = 0
    c = weights[0]
    duu = 1.0/N_resample
    r = np.random.uniform(0.0, 1.0) * duu
    
    for i in prange(N_resample):
        uu = r + i*duu
        while uu > c:
            idx += 1
            c += weights[idx]
        new_particle_poses[i] = particle_poses[idx]
        new_particle_beliefs[i] = particle_beliefs[idx]
    
    return new_particle_poses, new_particle_beliefs

@njit(cache = True)
def should_resample(weights : np.ndarray, 
                    steps_from_resample : int,
                    resample_steps_thresholds : np.ndarray) -> bool:
    '''
    input:
        weights - array of shape (N_particles)
        steps_from_resample - amount of particle filter steps since last resample
        resample_steps_thresholds - array of two elements.
            first element: minimum steps before resample can occur
            second element: amount of steps after which resample must occur

    output:
        True if we should resample, False otherwise
    '''
    N_particles = weights.shape[0]
    n_eff = weights.dot(weights)
    eta_threshold = 2.0/N_particles
    if n_eff < eta_threshold and steps_from_resample > resample_steps_thresholds[0] \
        or steps_from_resample >= resample_steps_thresholds[1]:
         return True
    else:
        return False