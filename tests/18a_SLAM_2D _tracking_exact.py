import numpy as np
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid, ScanSolid, ArrowSolid, TrailSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors.sensors import Lidar
from bim4loc.random.one_dim import Gaussian
from bim4loc.rbpf.tracking.exact import RBPF
from bim4loc.geometry.pose2z import compose_s, s_from_Rt, T_from_s
from bim4loc.random.multi_dim import gauss_fit
from bim4loc.existance_mapping import filters as existence_filters
from bim4loc.evaluation import evaluation
import matplotlib.pyplot as plt
import time
import logging
from copy import deepcopy
import keyboard
from numba import njit
from bim4loc.geometry.scan_matcher.scan_matcher import dead_reck_scan_match
from bim4loc.random.multi_dim import gauss_likelihood

np.random.seed(25)
logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

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
initial_beliefs = np.zeros(len(solids))
simulation_solids = []
z_prev = None
for i, s in enumerate(solids):
    s_simulation = s.clone()
    s_simulation_belief = s.schedule.cdf(current_time)
    s_simulation.set_existance_belief_and_shader(s_simulation_belief)
    
    initial_beliefs[i] = s_simulation_belief
    simulation_solids.append(s_simulation)
simulation = RayCastingMap(simulation_solids)

#INITALIZE DRONE AND SENSOR
drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar(angles_u = np.linspace(-np.pi/4,np.pi/4, int(300/4)), angles_v = np.array([0.0])); 
sensor.std = 0.1; sensor.piercing = False; sensor.max_range = 100.0
drone.mount_sensor(sensor)

simulated_sensor = deepcopy(sensor)
simulated_sensor.std = 5.0 * sensor.std
simulated_sensor.piercing = True

#BUILDING ACTION SET
straight = np.array([0.5,0.0 ,0.0 ,0.0])
turn_left = np.array([0.0 ,0.0 ,0.0, np.pi/8])
turn_right = np.array([0.0, 0.0, 0.0, -np.pi/8])
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20 + [turn_right] * 4

#SPREAD PARTICLES
bounds_min, bounds_max, extent = world.bounds()
N_particles = 10
particle_poses = np.vstack((np.random.normal(drone.pose[0], 0.2, N_particles),
                       np.random.normal(drone.pose[1], 0.2, N_particles),
                       np.full(N_particles,drone.pose[2]),
                       np.random.normal(drone.pose[3], np.radians(5.0), N_particles))).T
# N_particles = 1
# particle_poses = np.reshape(drone.pose, (1,4))
particle_beliefs = np.tile(initial_beliefs, (N_particles,1))
perfect_belief  = initial_beliefs.copy()

#initalize weights
weights = np.ones(N_particles) / N_particles
particle_reservoirs = np.zeros((N_particles, len(solids)))

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
# [visApp.add_solid(s,"simulation", f"{i}") for i,s in enumerate(simulation.solids)]
[visApp.add_solid(s,"simulation") for i,s in enumerate(simulation.solids)]
visApp.redraw("simulation")
visApp.show_axes(True,"simulation")
visApp.setup_default_camera("simulation")
vis_particles = ParticlesSolid(poses = particle_poses)
visApp.add_solid(vis_particles.lines, "simulation")
visApp.add_solid(vis_particles.tails, "simulation")
trail_est = TrailSolid("trail_est", drone.pose[:3].reshape(1,3))
visApp.add_solid(trail_est, "simulation")

#create initial map state window
visApp.add_scene("initial_state", "world")
[visApp.add_solid(s,"initial_state") for s in simulation.solids]
visApp.redraw("initial_state")
visApp.show_axes(True,"initial_state")
visApp.setup_default_camera("initial_state")
dead_reck = ArrowSolid("dead_reck", 1.0, drone.pose)
visApp.add_solid(dead_reck, "initial_state")
trail_dead_reck = TrailSolid("trail_dead_reck", drone.pose[:3].reshape(1,3))
visApp.add_solid(trail_dead_reck, "initial_state")

U_COV = np.diag([0.1, 0.05, 1e-25, np.radians(1.0)])/10

#create the sense_fcn
rbpf = RBPF(simulation, simulated_sensor, resample_rate = 4, U_COV = U_COV)

#history
mu, cov = gauss_fit(particle_poses.T, weights)
history = {'gt_traj': [drone.pose], 'perfect_beliefs': [initial_beliefs],
            'dead_reck' : [drone.pose],
           'est_traj': [mu], 'est_covs': [cov], 'est_beliefs': [initial_beliefs]}

#LOOP
time.sleep(2)
for t, u in enumerate(actions):
    # keyboard.wait('space')

    #move drone
    drone.move(u)
    
    #produce measurement
    z, _, _, z_p = drone.scan(world, project_scan = True, n_hits = 5, noisy = True)

    u_noisy = compose_s(u,np.random.multivariate_normal(np.zeros(4), U_COV))
    particle_poses, particle_beliefs, weights = rbpf.step(particle_poses, particle_beliefs, 
                                                          weights, particle_reservoirs,
                                                         u_noisy, U_COV, z)


    expected_map = np.sum(weights.reshape(-1,1) * particle_beliefs, axis = 0)
    best_map = particle_beliefs[np.argmax(weights)]
    expected_pose = np.sum(weights.reshape(-1,1) * particle_poses, axis = 0)

    #calculate dead reck
    if z_prev is not None:
        R,t, rmse = dead_reck_scan_match(T_from_s(u_noisy),z_prev, z, sensor.max_range,
                    sensor.get_scan_to_points(),
                    downsample_voxelsize = 0.5,
                    icp_distance_threshold = 10.0)
        pdf_scan_match = gauss_likelihood(s_from_Rt(R,t),np.zeros(4),U_COV)
        if rmse < 0.5: #and pdf_scan_match > 0.05: #downsample_voxelsize = 0.5
            dead_reck.pose = compose_s(dead_reck.pose, s_from_Rt(R.T,-R.T@t))
        else:
            dead_reck.pose = compose_s(dead_reck.pose, u_noisy)
    else:
        dead_reck.pose = compose_s(dead_reck.pose, u_noisy)
    z_prev = z
    dead_reck.update_geometry(dead_reck.pose)
    
    #updating drawings
    simulation.update_solids_beliefs(best_map)        
    vis_scan.update(drone.pose[:3], z_p.T)
    vis_particles.update(particle_poses, weights)
    visApp.update_solid(vis_scan)
    visApp.update_solid(drone.solid)
    visApp.update_solid(vis_particles.lines, "simulation")
    visApp.update_solid(vis_particles.tails, "simulation")
    [visApp.update_solid(s,"simulation") for s in simulation.solids]
    trail_est.update(expected_pose[:3].reshape(1,-1))
    visApp.update_solid(trail_est, "simulation")
    visApp.update_solid(dead_reck, "initial_state")
    trail_dead_reck.update(dead_reck.pose[:3].reshape(1,-1))
    visApp.update_solid(trail_dead_reck, "initial_state")
    visApp.redraw_all_scenes()

    #calculate perfect mapping with known poses
    perfect_simulated_z, perfect_simulated_z_ids, _, _, _ = simulated_sensor.sense(drone.pose, simulation, 5, noisy = False)
    existence_filters.exact(perfect_belief, z, 
                            perfect_simulated_z, perfect_simulated_z_ids, 
                            simulated_sensor.std, simulated_sensor.max_range)

    #log history
    history['gt_traj'].append(drone.pose)
    history['dead_reck'].append(dead_reck.pose)
    mu, cov = gauss_fit(particle_poses.T, weights)
    history['est_traj'].append(mu)
    history['est_covs'].append(cov)
    history['est_beliefs'].append(expected_map)
    history['perfect_beliefs'].append(perfect_belief.copy())

    # time.sleep(0.1)

#run simulation of perfect mapping
evaluation.localiztion_error(np.array(history['gt_traj']), 
                             np.array(history['est_traj']),
                             np.array(history['est_covs']),
                             np.array(history['dead_reck']))
evaluation.map_entropy(np.array(history['est_beliefs']),
                 np.array(history['perfect_beliefs']))
ground_truth_beliefs = np.array([1.0 if s.name in [c.name for c in constructed_solids] else 0.0 for s in solids])
evaluation.cross_entropy_error(ground_truth_beliefs,
                                np.array(history['est_beliefs']),
                                np.array(history['perfect_beliefs']))
plt.show()