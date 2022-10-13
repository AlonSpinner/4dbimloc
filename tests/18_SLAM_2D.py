from tkinter import N
import numpy as np
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ScanSolid, ParticlesSolid
from bim4loc.agents import Drone
from bim4loc.sensors.sensors import Lidar
from bim4loc.maps import RayCastingMap
from bim4loc.geometry.pose2z import compose_s
from bim4loc.random.one_dim import Gaussian
from bim4loc.existance_mapping.filters import exact, approx
from bim4loc.random.utils import p2logodds, logodds2p
import time
import logging
import copy
import keyboard

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.INFO)

#FUNCTIONS
gaussian_pdf = Gaussian._pdf

#BUILD WORLD
current_time = 5.0 #[s]
solids = ifc_converter(IFC_PATH)

constructed_solids = []
for s in solids:
    s.set_random_completion_time()
    if s.completion_time < current_time:
        constructed_solids.append(s.clone())
world = RayCastingMap(constructed_solids)

#BUILD SIMULATION
beliefs = np.zeros(len(solids))
simulation_solids = []
for i, s in enumerate(solids):
    s_simulation = s.clone()
    s_simulation_belief = s.schedule.cdf(current_time)
    s_simulation.set_existance_belief_and_shader(s_simulation_belief)
    
    beliefs[i] = p2logodds(s_simulation_belief)
    # beliefs[i] = s_simulation_belief
    simulation_solids.append(s_simulation)
simulation = RayCastingMap(simulation_solids)

#INITALIZE DRONE AND SENSOR
drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar(angles_u = np.linspace(-np.pi/2,np.pi/2, 8), angles_v = np.array([0.0])); 
sensor.std = 0.1; sensor.piercing = False; sensor.max_range = 100.0
drone.mount_sensor(sensor)

simulated_sensor = copy.deepcopy(sensor)
simulated_sensor.std = 5.0 * sensor.std
simulated_sensor.piercing = True

#BUILDING ACTION SET
straight = np.array([0.5,0.0 ,0.0 ,0.0])
turn_left = np.array([0.0 ,0.0 ,0.0, np.pi/8])
turn_right = np.array([0.0, 0.0, 0.0, -np.pi/8])
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

#SPREAD PARTICLES UNIFORMLY
bounds_min, bounds_max, extent = world.bounds()
N_particles = 200

# particle_poses = np.vstack((np.random.uniform(bounds_min[0], bounds_max[0], N_particles),
#                        np.random.uniform(bounds_min[1], bounds_max[1], N_particles),
#                        np.full(N_particles,drone.pose[2]),
#                        np.random.uniform(-np.pi, np.pi, N_particles))).T

particle_poses = np.vstack((np.random.normal(drone.pose[0], 0.5, N_particles),
                       np.random.normal(drone.pose[1], 0.5, N_particles),
                       np.full(N_particles,drone.pose[2]),
                       np.random.normal(drone.pose[3], np.radians(7.0), N_particles))).T

particle_beliefs = np.tile(beliefs, (N_particles,1))

#INITALIZE WEIGHTS
weights = np.ones(N_particles) / N_particles


#DRAW
visApp = VisApp()
[visApp.add_solid(s,"world") for s in world.solids]
visApp.redraw("world")
visApp.show_axes(True,"world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")
vis_scan = ScanSolid("scan")
visApp.add_solid(vis_scan, "world")
#create simulation window
visApp.add_scene("simulation", "world")
[visApp.add_solid(s,"simulation") for s in simulation.solids]
visApp.redraw("simulation")
visApp.show_axes(True,"simulation")
visApp.setup_default_camera("simulation")
vis_particles = ParticlesSolid(poses = particle_poses)
visApp.add_solid(vis_particles.lines, "simulation")
visApp.add_solid(vis_particles.tails, "simulation")

U_COV = np.diag([0.05, 0.05, 0.0, np.radians(1.0)])/100
ETA_THRESHOLD = 5.0/N_particles
ALPHA_SLOW = 0* 0.001 #0.0 <= ALPHA_SLOW << ALPHA_FAST, also: http://wiki.ros.org/amcl
ALPHA_FAST = 0* 2.0
POSE_MIN_BOUNDS = np.array([bounds_min[0],bounds_min[1], 0.0 , -np.pi])
POSE_MAX_BOUNDS = np.array([bounds_max[0],bounds_max[1], 0.0 , np.pi])
MIN_STEPS_4_RESAMPLE = 3
CEILING_STEPS_4_RESAMPLE = 5
w_slow = w_fast = 0.0
#LOOP
time.sleep(2)
steps_from_resample = CEILING_STEPS_4_RESAMPLE
# keyboard.wait('space')
for t, u in enumerate(actions):
    #add 1 more step
    steps_from_resample += 1

    #move drone
    drone.move(u)
    
    #produce measurement
    z, z_ids, z_normals, z_p = drone.scan(world, project_scan = True, 
                                                 noisy = True, 
                                                 n_hits = 5)

    #---------------------------FILTER-------------------------------------
    #compute weights and normalize
    sum_weights = 0.0
    noisy_u = np.random.multivariate_normal(u, U_COV, N_particles)
    for i in range(N_particles):
        particle_poses[i] = compose_s(particle_poses[i], noisy_u[i])

        if np.any(particle_poses[i][:3] < bounds_min[:3]) \
             or np.any(particle_poses[i][:3] > bounds_max[:3]):
            weights[i] = 0.0
            continue

        particle_z_values, particle_z_ids, _, particle_z_cos_incident, _ \
            = simulated_sensor.sense(particle_poses[i], 
                                     simulation, n_hits = 5, 
                                     noisy = False)
        
        particle_stds = simulated_sensor.std#/ np.abs(particle_z_cos_incident)
        # pz = np.max(gaussian_pdf(particle_z_values, particle_stds, z.reshape(-1,1), pseudo = True), axis = 1)
        pz = 0.1 + 0.9 * gaussian_pdf(particle_z_values, particle_stds, z.reshape(-1,1), pseudo = True)
        
        #line 205 in https://github.com/ros-planning/navigation/blob/noetic-devel/amcl/src/amcl/sensors/amcl_laser.cpp
        weights[i] *= 1.0 + np.sum(pz**3)
        # weights[i] *= np.product(pz)
        
        sum_weights += weights[i]

            #update mapping
        approx(particle_beliefs[i], 
            z, 
            particle_z_values, 
            particle_z_ids, 
            simulated_sensor.std, 
            simulated_sensor.max_range)
    
    if sum_weights == 0.0:
        weights = np.ones(N_particles) / N_particles

    else:
        #normalize
        weights = weights / sum_weights

        #Updating w_slow and w_fast
        w_avg = sum_weights / N_particles

        if w_slow == 0.0:
            w_slow = w_avg
        else:
            w_slow = w_slow + ALPHA_SLOW * (w_avg - w_slow)

        if w_fast == 0.0:
            w_fast = w_avg
        else:        
            w_fast = w_fast + ALPHA_FAST * (w_avg - w_fast)

    #resample
    # https://github.com/ros-planning/navigation/blob/noetic-devel/amcl/src/amcl/pf/pf.c
    # "void pf_update_resample"
    n_eff = weights.dot(weights)
    if n_eff < ETA_THRESHOLD and steps_from_resample >= MIN_STEPS_4_RESAMPLE \
        or steps_from_resample >= CEILING_STEPS_4_RESAMPLE:
        steps_from_resample = 0
        
        w_diff = np.clip(1.0 - w_fast / w_slow, 0.0, 1.0) #percentage of random samples

        logging.info(f"resampling with w_diff = {w_diff}")
        
        N_random = int(w_diff * N_particles)
        if N_random > 0:
            random_samples = np.random.uniform(POSE_MIN_BOUNDS, POSE_MAX_BOUNDS, (N_random , 4))
        
        N_resample = N_particles - N_random
        if N_resample > 0:
            resample_samples = np.zeros((N_resample,4))
            resample_beliefs = np.zeros((N_resample,len(simulation.solids)))
            idx = 0
            c = weights[0]
            duu = 1.0/N_resample
            r = np.random.uniform() * duu
            for i in range(N_resample):
                uu = r + i*duu
                while uu > c:
                    idx += 1
                    c += weights[idx]
                resample_samples[i] = particle_poses[idx]
                resample_beliefs[i] = particle_beliefs[idx]

            # particle_beliefs_4new = particle_beliefs[np.argmax(weights)]
            particle_beliefs_4new = np.sum(weights.reshape(-1,1) * particle_beliefs, axis = 0)

        if N_resample > 0 and N_random > 0:

            particle_poses = np.vstack((random_samples, resample_samples))
            particle_beliefs = np.vstack((np.tile(particle_beliefs_4new, (N_random,1)),
                                           resample_beliefs))
        elif N_random > 0:
            particle_poses = random_samples
            particle_beliefs = np.tile(particle_beliefs_4new, (N_random,1))
        else:
            particle_poses = resample_samples
            particle_beliefs = resample_beliefs

        weights = np.ones(N_particles) / N_particles

        #Reset averages, to avoid spiraling off into complete randomness.
        if w_diff > 0.0:
            w_slow = w_fast = 0.0

    #updating drawings
    estimate_beliefs = np.sum(weights.reshape(-1,1) * particle_beliefs, axis = 0)
    # estimate_beliefs = particle_beliefs[np.argmax(weights)]
    simulation.update_solids_beliefs(logodds2p(estimate_beliefs))
    # simulation.update_solids_beliefs(estimate_beliefs)
    vis_scan.update(drone.pose[:3], z_p.T)
    vis_particles.update(particle_poses, weights)
    visApp.update_solid(vis_scan)
    visApp.update_solid(drone.solid)
    visApp.update_solid(vis_particles.lines, "simulation")
    visApp.update_solid(vis_particles.tails, "simulation")
    [visApp.update_solid(s,"simulation") for s in simulation.solids]
    visApp.redraw_all_scenes()

    # time.sleep(0.01)
    # keyboard.wait('space')
    
print('finished')
