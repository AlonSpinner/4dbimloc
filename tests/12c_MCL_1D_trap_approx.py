import numpy as np
from bim4loc.geometry.poses import Pose2z
from bim4loc.binaries.paths import IFC_LINEUP_TRAP_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid, ScanSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors import Lidar
from bim4loc.random.one_dim import Gaussian
from bim4loc.existance_mapping.filters import exact, approx
from bim4loc.random.utils import logodds2p, p2logodds
import time
import logging
from copy import deepcopy
import matplotlib.pyplot as plt

np.random.seed(25)
logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

#FUNCTIONS
gaussian_pdf = Gaussian._pdf

#BUILD WORLD
solids = ifc_converter(IFC_PATH)
constructed_solids = [s.clone() for s in solids[:-2]]

beliefs = 1.0 * np.ones(len(solids))
beliefs[[-1, -2, 6, 10]] = 0.5
for i, b in enumerate(beliefs):
    solids[i].set_existance_belief_and_shader(b)

simulation = RayCastingMap(solids)
world = RayCastingMap(constructed_solids)

#INITALIZE DRONE AND SENSOR
drone = Drone(pose = Pose2z(3.0 ,10.0 ,0.0 , 1.5))
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
N_particles = 100
particle_poses = []
for i in range(N_particles):
    particle_poses.append(
        Pose2z(3.0,
                np.random.uniform(bounds_min[1], bounds_max[1]),
                0.0,
                0.0)
    )

particle_beliefs = p2logodds(np.tile(beliefs, (N_particles,1)))

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

u = np.array([0.0 ,0.2 ,0.0 ,0.0])
U_COV = np.diag([0.0, 0.02, 0.0, 0.0])
#LOOP
time.sleep(2)
for t in range(100):
    #move drone
    drone.move(u)
    
    #produce measurement
    z, z_ids, z_normals, z_p = drone.scan(world, project_scan = True)

    #---------------------------FILTER-------------------------------------
    #compute weights and normalize
    sum_weights = 0.0
    for i in range(N_particles):
        particle_poses[i] = particle_poses[i].retract(np.random.multivariate_normal(u, U_COV))
        particle_z_values, particle_z_ids, _ = simulated_sensor.sense(particle_poses[i], 
                                                                    simulation, n_hits = 2, 
                                                                    noisy = False)
        
        # exact(particle_beliefs[i], z, particle_z_values, particle_z_ids, simulated_sensor.std, simulated_sensor.max_range)
        approx(particle_beliefs[i], 
                z, 
                particle_z_values, 
                particle_z_ids, 
                simulated_sensor.std, 
                simulated_sensor.max_range)

        pz = 0.2 + 0.8 * gaussian_pdf(particle_z_values, simulated_sensor.std, z, pseudo = True)
        weights[i] = weights[i] * np.max(pz)
        sum_weights += weights[i]
    #normalize
    weights = weights / sum_weights
    
    #resample
    if (t % 5) == 0:
        r = np.random.uniform()/N_particles
        idx = 0
        c = weights[idx]
        new_particle_poses = []
        new_particle_beliefs = np.zeros_like(particle_beliefs)
        for i in range(N_particles):
            uu = r + i*1/N_particles
            while uu > c:
                idx += 1
                c += weights[idx]
            new_particle_poses.append(deepcopy(particle_poses[idx]))
            new_particle_beliefs[i] = particle_beliefs[idx]
        particle_poses = new_particle_poses
        particle_beliefs = new_particle_beliefs
        weights = np.ones(N_particles) / N_particles
        
        # print('resampled')

    if t % 5 != 0:
        # best_particle_idx = np.argmax(weights)
        # simulation.update_solids_beliefs(logodds2p(particle_beliefs[best_particle_idx]))
        estimate_beliefs = np.sum(weights.reshape(-1,1) * logodds2p(particle_beliefs), axis = 0)
        simulation.update_solids_beliefs(estimate_beliefs)        

    #updating drawings
    vis_scan.update(drone.pose.t, z_p)
    visApp.update_solid(drone.solid, "world")
    visApp.update_solid(vis_scan, "world")
    vis_particles.update(particle_poses)
    visApp.update_solid(vis_particles.lines, "simulation")
    visApp.update_solid(vis_particles.tails, "simulation")
    [visApp.update_solid(s,"simulation") for s in simulation.solids]

    # plt.scatter([p.t[1] for p in particle_poses], weights)
    # plt.xlim([bounds_min[1], bounds_max[1]])
    # plt.show()

    # time.sleep(0.1)