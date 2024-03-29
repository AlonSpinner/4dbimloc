from numba import njit, prange
import numpy as np
from typing import Tuple

# @njit(cache = True)
def low_variance_sampler(weights : np.ndarray, 
                         particle_poses : np.ndarray,
                         particle_beliefs : np.ndarray, 
                         N_resample : int,
                         particle_reservoirs : np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
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
    if particle_reservoirs is None:
        new_particle_reservoirs = None
    else:
        new_particle_reservoirs = np.zeros((N_resample, particle_reservoirs.shape[1]))
    
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
        if particle_reservoirs is not None:
            new_particle_reservoirs[i] = particle_reservoirs[idx]
    
    return new_particle_poses, new_particle_beliefs, new_particle_reservoirs