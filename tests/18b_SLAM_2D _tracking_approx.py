import numpy as np
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid, ScanSolid, ArrowSolid, TrailSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors.sensors import Lidar
from bim4loc.random.one_dim import Gaussian
from bim4loc.rbpf.tracking.approx import RBPF
from bim4loc.geometry.pose2z import compose_s
import time
import logging
from copy import deepcopy
import keyboard
from bim4loc.evaluation import evaluation
from bim4loc.random.multi_dim import gauss_fit
import matplotlib.pyplot as plt

np.random.seed(25)
logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

#FUNCTIONS
gaussian_pdf = Gaussian._pdf

#BUILD WORLD
current_time = 5.0 #[s]
solids = ifc_converter(IFC_PATH)
# solids = solids[:10]

constructed_solids = []
for s in solids:
    s.set_random_completion_time()
    if s.completion_time < current_time:
        constructed_solids.append(s.clone())
world = RayCastingMap(constructed_solids)

#BUILD SIMULATION
initial_beliefs = np.zeros(len(solids))
simulation_solids = []
for i, s in enumerate(solids):
    s_simulation = s.clone()
    s_simulation_belief = s.schedule.cdf(current_time)
    s_simulation.set_existance_belief_and_shader(s_simulation_belief)
    
    initial_beliefs[i] = s_simulation_belief
    simulation_solids.append(s_simulation)
simulation = RayCastingMap(simulation_solids)

#INITALIZE DRONE AND SENSOR
drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar(angles_u = np.linspace(-np.pi,np.pi, 300), angles_v = np.array([0.0])); 
sensor.std = 0.1; sensor.piercing = False; sensor.max_range = 100.0
drone.mount_sensor(sensor)

simulated_sensor = deepcopy(sensor)
simulated_sensor.std = 5.0 * sensor.std
simulated_sensor.piercing = True

#BUILDING ACTION SET
straight = np.array([0.5,0.0 ,0.0 ,0.0])
turn_left = np.array([0.0 ,0.0 ,0.0, np.pi/8])
turn_right = np.array([0.0, 0.0, 0.0, -np.pi/8])
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

#SPREAD PARTICLES
bounds_min, bounds_max, extent = world.bounds()
N_particles = 10

particle_poses = np.vstack((np.random.normal(drone.pose[0], 0.2, N_particles),
                       np.random.normal(drone.pose[1], 0.2, N_particles),
                       np.full(N_particles,drone.pose[2]),
                       np.random.normal(drone.pose[3], np.radians(5.0), N_particles))).T
particle_beliefs = np.tile(initial_beliefs, (N_particles,1))

#initalize weights
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

U_COV = np.diag([0.05, 0.05, 0.0, np.radians(1.0)])/10
map_bounds_min, map_bounds_max, extent = simulation.bounds()

#create the sense_fcn
sense_fcn = lambda x: simulated_sensor.sense(x, simulation, n_hits = 5, noisy = False)
rbpf = RBPF(sense_fcn, simulated_sensor.get_scan_to_points(),
            simulated_sensor.std, simulated_sensor.max_range,
            map_bounds_min, map_bounds_max, resample_rate = 3)

#history
mu, cov = gauss_fit(particle_poses.T, weights)
history = {'gt_traj': [drone.pose], 'dead_reck' : [drone.pose],
           'est_traj': [mu], 'est_covs': [cov], 'est_beliefs': []}

#LOOP
time.sleep(2)
for t, u in enumerate(actions):
    # keyboard.wait('space')

    #move drone
    drone.move(u)
    
    #produce measurement
    z, _, _, z_p = drone.scan(world, project_scan = True, n_hits = 5, noisy = True)

    u_noisy = compose_s(np.zeros(4),np.random.multivariate_normal(u, U_COV))
    particle_poses, particle_beliefs, weights = rbpf.step(particle_poses, particle_beliefs, weights,
                                                         u, U_COV, z)

    if (t % 2) != 0:
        expected_map = np.sum(weights.reshape(-1,1) * particle_beliefs, axis = 0)
        best_map = particle_beliefs[np.argmax(weights)]
        simulation.update_solids_beliefs(best_map)        
    
    #updating drawings
    vis_scan.update(drone.pose[:3], z_p.T)
    vis_particles.update(particle_poses, weights)
    visApp.update_solid(vis_scan)
    visApp.update_solid(drone.solid)
    visApp.update_solid(vis_particles.lines, "simulation")
    visApp.update_solid(vis_particles.tails, "simulation")
    [visApp.update_solid(s,"simulation") for s in simulation.solids]

    dead_reck.update_geometry(compose_s(dead_reck.pose, u_noisy))
    visApp.update_solid(dead_reck, "initial_state")
    trail_dead_reck.update(dead_reck.pose[:3].reshape(1,-1))
    visApp.update_solid(trail_dead_reck, "initial_state")
    visApp.redraw_all_scenes()

    #log history
    history['gt_traj'].append(drone.pose)
    history['dead_reck'].append(dead_reck.pose)
    mu, cov = gauss_fit(particle_poses.T, weights)
    history['est_traj'].append(mu)
    history['est_covs'].append(cov)

    # time.sleep(0.1)

#run simulation of perfect mapping
evaluation.localiztion_error(np.array(history['gt_traj']), 
                             np.array(history['est_traj']),
                             np.array(history['est_covs']),
                             np.array(history['dead_reck']))
plt.show()