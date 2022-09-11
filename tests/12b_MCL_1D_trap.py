import numpy as np
from bim4loc.geometry.poses import Pose2z
from bim4loc.binaries.paths import IFC_LINEUP_TRAP_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid, ScanSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors import Lidar
from bim4loc.random.one_dim import Gaussian
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
for s in solids:
    s.set_existance_belief_and_shader(1.0)
solids[6].set_existance_belief_and_shader(0.5)
solids[10].set_existance_belief_and_shader(0.5)
solids[-1].set_existance_belief_and_shader(0.5)
solids[-2].set_existance_belief_and_shader(0.5)

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
particles = []
for i in range(N_particles):
    particles.append(
        Pose2z(3.0,
                np.random.uniform(bounds_min[1], bounds_max[1]),
                0.0,
                0.0)
    )
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
vis_particles = ParticlesSolid(poses = particles)
visApp.add_solid(vis_particles.lines, "simulation")
visApp.add_solid(vis_particles.tails, "simulation")

u = np.array([0.0 ,0.2 ,0.0 ,0.0])
U_COV = np.diag([0.0, 0.02, 0.0, 0.0])
#LOOP
for t in range(100):
    #move drone
    drone.move(u)
    
    #produce measurement
    z, z_ids, z_normals, z_p = drone.scan(world, project_scan = True)

    #---------------------------FILTER0-------------------------------------
    #compute weights and normalize
    sum_weights = 0.0
    for i in range(N_particles):
        particles[i] = particles[i].retract(np.random.multivariate_normal(u, U_COV))
        particle_z_values, particle_z_ids, _ = simulated_sensor.sense(particles[i], 
                                                                    simulation, n_hits = 2, 
                                                                    noisy = False)
        
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
        new_particles = []
        for i in range(N_particles):
            uu = r + i*1/N_particles
            while uu > c:
                idx += 1
                c += weights[idx]
            new_particles.append(deepcopy(particles[idx]))
        particles = new_particles
        weights = np.ones(N_particles) / N_particles
        # print('resampled')

    #updating drawings
    vis_scan.update(drone.pose.t, z_p)
    visApp.update_solid(drone.solid, "world")
    visApp.update_solid(vis_scan, "world")
    vis_particles.update(particles)
    visApp.update_solid(vis_particles.lines, "simulation")
    visApp.update_solid(vis_particles.tails, "simulation")

    # plt.scatter([p.t[1] for p in particles], weights)
    # plt.xlim([bounds_min[1], bounds_max[1]])
    # plt.show()

    # time.sleep(0.1)