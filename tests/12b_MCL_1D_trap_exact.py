import numpy as np
from bim4loc.binaries.paths import IFC_LINEUP_TRAP_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid, ScanSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors.sensors import Lidar
from bim4loc.random.one_dim import Gaussian
from bim4loc.fast_slam.exact_filter import fast_slam_lpf_resampler
import time
import logging
from copy import deepcopy
import keyboard
from numba import njit

np.random.seed(25) #25, 24 are bad. 23 looks good :X
logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

#FUNCTIONS
gaussian_pdf = Gaussian._pdf

#BUILD WORLD
solids = ifc_converter(IFC_PATH)
constructed_solids = [s.clone() for s in solids[:-2]]

initial_beliefs = np.ones(len(solids))
initial_beliefs[[-1, -2, 6, 10]] = 0.5
# beliefs[[-1, -2]] = 1.0
for i, b in enumerate(initial_beliefs):
    solids[i].set_existance_belief_and_shader(b)

simulation = RayCastingMap(solids)
world = RayCastingMap(constructed_solids)

#INITALIZE DRONE AND SENSOR
drone = Drone(pose = np.array([3.0 ,10.0 ,1.5 , 0.0]))
sensor = Lidar(angles_u = np.array([0.0]),
                 angles_v = np.array([0.0])); 
sensor.std = 0.1
sensor.piercing = False
sensor.max_range = 100.0
drone.mount_sensor(sensor)

simulated_sensor = deepcopy(sensor)
simulated_sensor.piercing = True

#SPREAD PARTICLES UNIFORMLY
bounds_min, bounds_max, _ = world.bounds()
N_particles = 200
particle_poses = np.vstack((np.full(N_particles, 3.0),
                       np.random.uniform(bounds_min[1], bounds_max[1], N_particles),
                       np.zeros(N_particles),
                       np.full(N_particles, 0.0))).T
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

u = np.array([0.0 ,0.2 ,0.0 ,0.0])
U_COV = np.diag([0.0, 0.02, 0.0, 0.0])
steps_from_resample = 0
w_slow = w_fast = 0.0
map_bounds_min = np.array([0.0, 0.0, 0.0]) #filler values
map_bounds_max = np.array([10.0, 10.0, 0.0]) #filler values


#create the sense_fcn
sense_fcn = lambda x: simulated_sensor.sense(x, simulation, n_hits = 10, noisy = False)

#LOOP
time.sleep(2)
for t in range(100):
    # keyboard.wait('space')

    #move drone
    drone.move(u)
    
    #produce measurement
    z, _, _, z_p = drone.scan(world, project_scan = True)

    particle_poses, particle_beliefs, \
    weights, w_slow, w_fast, w_diff, steps_from_resample = \
         fast_slam_lpf_resampler(particle_poses, particle_beliefs, weights, u, U_COV, z, 
                    steps_from_resample, w_slow, w_fast,
                    sense_fcn, simulated_sensor.std, simulated_sensor.max_range, 
                    map_bounds_min, map_bounds_max, initial_beliefs,
                    resample_steps_thresholds = np.array([1,2]))

    if (t % 2) != 0:
        estimate_beliefs = np.sum(weights.reshape(-1,1) * particle_beliefs, axis = 0)
        simulation.update_solids_beliefs(estimate_beliefs)        
    #updating drawings
    vis_scan.update(drone.pose[:3], z_p.T)
    vis_particles.update(particle_poses, weights)
    visApp.update_solid(vis_scan)
    visApp.update_solid(drone.solid)
    visApp.update_solid(vis_particles.lines, "simulation")
    visApp.update_solid(vis_particles.tails, "simulation")
    [visApp.update_solid(s,"simulation") for s in simulation.solids]

    # time.sleep(0.1)