from numba import njit, prange
from bim4loc.geometry.pose2z import compose_s
from bim4loc.sensors.models import inverse_lidar_model
from bim4loc.random.utils import p2logodds,logodds2p
from bim4loc.existance_mapping.filters import approx
import numpy as np


def low_variance_sampler(weights, particle_poses, particle_beliefs):
    N_particles = weights.shape[0]

    r = np.random.uniform()/N_particles
    idx = 0
    c = weights[idx]
    new_particle_poses = np.zeros_like(particle_poses)
    new_particle_beliefs = np.zeros_like(particle_beliefs)
    
    for i in range(N_particles):
        uu = r + i*1/N_particles
        while uu > c:
            idx += 1
            c += weights[idx]
        new_particle_poses[i] = particle_poses[idx]
        new_particle_beliefs[i] = particle_beliefs[idx]
    
    particle_poses = new_particle_poses
    particle_beliefs = new_particle_beliefs
    return particle_poses, particle_beliefs

def fast_slam_filter(weights, particle_poses, particle_beliefs, u, U_COV, z, 
                    sense_fcn, lidar_std, lidar_max_range, 
                    map_bounds_min, map_bounds_max):
    
    N_particles = particle_poses.shape[0]

    #compute weights and normalize
    sum_weights = 0.0
    noisy_u = np.random.multivariate_normal(u, U_COV, N_particles)
    for k in prange(N_particles):
        #move
        particle_poses[k] = compose_s(particle_poses[k], noisy_u[k])
        #if particle moved outside the map, redraw a new one?
        if np.any(particle_poses[k][:3] < map_bounds_min[:3]) \
             or np.any(particle_poses[k][:3] > map_bounds_max[:3]):
            weights[k] = 0.0
            continue

        #calcualte weight
        particle_z_values, particle_z_ids, _, _, _ = sense_fcn(particle_poses[k])
        pz = np.zeros(len(z))
        for i in prange(len(z)):
            _, pz[i] = inverse_lidar_model(z[i], particle_z_values[i], particle_z_ids[i], particle_beliefs[k], 
                            lidar_std, lidar_max_range)
            
        weights[k] *= np.product(pz)
        sum_weights += weights[k]

        #remap 
        logodds_particle_beliefs = approx(p2logodds(particle_beliefs[i]), 
                                    z, 
                                    particle_z_values, 
                                    particle_z_ids, 
                                    lidar_std, 
                                    lidar_max_range)
        particle_beliefs[i] = logodds2p(logodds_particle_beliefs)

    if sum_weights == 0.0:
        weights = np.ones(N_particles) / N_particles
    weights = weights / sum_weights

    #Updating w_slow and w_fast
    w_avg = sum_weights / N_particles

    #update adaptive part of the filter
    if w_slow == 0.0:
        w_slow = w_avg
    else:
        w_slow = w_slow + ALPHA_SLOW * (w_avg - w_slow)
    if w_fast == 0.0:
        w_fast = w_avg
    else:        
        w_fast = w_fast + ALPHA_FAST * (w_avg - w_fast)

    #resample
    n_eff = weights.dot(weights)

            

    
