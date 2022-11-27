from bim4loc.rbpf.utils import low_variance_sampler
from bim4loc.random.multi_dim import sample_uniform
from typing import Tuple
from numba import njit
import numpy as np

@njit(cache = True)
def update_resampling_ws(w_slow : float, w_fast : float,
                         sum_weights : float, N_particles : int,
                         alpha_slow = 0.001, alpha_fast = 2.0,) \
                         -> Tuple[float, float]:
    '''
    here we have two low pass filters, one fast and one slow.
    we later test the ratio between the two to see 
    if there was a quick change in the average weights for the worst.
    if there was, we will throw random particles from thin air into the mix

    input:
        w_slow, w_fast - filtered variables
        sum_weights - sum of weights of all particles before resampling
        N_particles - number of particles
        alpha_slow, alpha_fast - low pass filter constants

    output:
        new w_slow and w_fast
    '''
    w_avg = sum_weights / N_particles
    if w_slow == 0.0: #happens after filter resampling
        w_slow = w_avg
    else:
        w_slow = w_slow + alpha_slow * (w_avg - w_slow)
    if w_fast == 0.0: #happens after filter resampling
        w_fast = w_avg
    else:        
        w_fast = w_fast + alpha_fast * (w_avg - w_fast)
    
    return w_slow, w_fast

@njit(cache = True)
def resampler(particle_poses : np.ndarray,
                particle_beliefs : np.ndarray,
                weights : np.ndarray,
                w_slow : float, w_fast : float,
                pose_min_bounds : np.ndarray, pose_max_bounds : np.ndarray, 
                initial_belief : np.ndarray):
    '''
    called in the case resampling is needed

    input: 
        particle_poses - array of shape (N_particles, 4)
        particle_beliefs - array of shape (N_particles, N_cells)
        weights - array of shape (N_particles)
        w_slow, w_fast - floats for adaptive resampling
        pose_min_bounds, pose_max_bounds - arrays of shape (4)
        initial_belief - array of shape (N_cells)
    
    output:
        particle_poses - updated, 
        particle_beliefs - updated
        weights - updated, 
        w_slow, w_fast - updated, 
        w_diff - 0.0 if we have good samples, 1.0 if samples are trash
    '''
    N_particles = particle_poses.shape[0]
    
    #percentage of random samples
    w_diff = 1.0 - w_fast / w_slow
    if w_diff > 1.0: #clip it
        w_diff = 1.0
    elif w_diff < 0.0:
        w_diff = 0.0
    N_random = int(w_diff * N_particles)
    N_resample = N_particles - N_random
    
    N_resample = N_particles #for now, we will resample all particles
    N_random = 0

    #produce new samples from static distribuion with initial belief maps
    if N_random > 0:
        thinAir_particle_poses = sample_uniform(pose_min_bounds, pose_max_bounds, N_random)
        thinAir_particle_beliefs = initial_belief.repeat(N_random).reshape((-1, N_random)).T

    #produce resampled samples                                             
    if N_resample > 0:
        resampled_particle_poses, resampled_particle_beliefs = \
            low_variance_sampler(weights, particle_poses, particle_beliefs, N_resample)
    
    #mash them together
    if N_random == 0:
        particle_poses = resampled_particle_poses
        particle_beliefs = resampled_particle_beliefs
    elif N_resample == 0:
        particle_poses = thinAir_particle_poses
        particle_beliefs = thinAir_particle_beliefs
    else:
        particle_poses = np.vstack((thinAir_particle_poses, resampled_particle_poses))
        particle_beliefs = np.vstack((thinAir_particle_beliefs, resampled_particle_beliefs))

    #reset weights
    weights = np.ones(N_particles) / N_particles
    #reset averages, to avoid spiraling off into complete randomness.
    if w_diff > 0.0:
        w_slow = w_fast = 0.0
    
    return particle_poses, particle_beliefs, weights, w_slow, w_fast, w_diff