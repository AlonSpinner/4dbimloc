import numpy as np
from bim4loc.binaries.paths import IFC_LINEUP_PATH as IFC_PATH
from bim4loc.geometry.pose2z import compose_s
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
world = RayCastingMap(solids)

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
N_particles = 100
particles = np.vstack((np.full(N_particles, 3.0),
                       np.random.uniform(bounds_min[1], bounds_max[1], N_particles),
                       np.zeros(N_particles),
                       np.full(N_particles, 0.0))).T

#initalize weights
weights = np.ones(N_particles) / N_particles

#DRAW
visApp = VisApp()
[visApp.add_solid(s,"world") for s in world.solids]
visApp.redraw("world")
visApp.show_axes(True,"world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")
vis_particles = ParticlesSolid(poses = particles)
visApp.add_solid(vis_particles.lines, "world")
visApp.add_solid(vis_particles.tails, "world")
vis_scan = ScanSolid("scan")
visApp.add_solid(vis_scan, "world")

u = np.array([0.0 ,0.2 ,0.0 ,0.0])
U_COV = np.diag([0.0, 0.02, 0.0, 0.0])
#LOOP
for t in range(100):
    #move drone
    drone.move(u)
    
    #produce measurement
    z, z_ids, z_normals, z_p = drone.scan(world, project_scan = True, noisy = True)

    #---------------------------FILTER-------------------------------------
    #compute weights and normalize
    sum_weights = 0.0
    noisy_u = np.random.multivariate_normal(u, U_COV, N_particles)
    for i in range(N_particles):
        particles[i] = compose_s(particles[i], noisy_u[i])
        particle_z_values, particle_z_ids, _ = sensor.sense(particles[i], 
                                                                    world, n_hits = 10, 
                                                                    noisy = False)
        
        pz = 0.2 + 0.8 * gaussian_pdf(particle_z_values, sensor.std, z, pseudo = True)
        weights[i] *= np.product(pz)
        sum_weights += weights[i]
    #normalize
    weights = weights / sum_weights
    
    #resample
    n_eff = weights.dot(weights)
    if t % 5 == 0:
        r = np.random.uniform()/N_particles
        idx = 0
        c = weights[idx]
        new_particles = np.zeros_like(particles)
        for i in range(N_particles):
            uu = r + i*1/N_particles
            while uu > c:
                idx += 1
                c += weights[idx]
            new_particles[i] = particles[idx]
        particles = new_particles
        weights = np.ones(N_particles) / N_particles

    #updating drawings
    vis_scan.update(drone.pose[:3], z_p.T)
    vis_particles.update(particles)
    visApp.update_solid(vis_scan)
    visApp.update_solid(drone.solid)
    visApp.update_solid(vis_particles.lines)
    visApp.update_solid(vis_particles.tails)

    # plt.scatter([p.t[1] for p in particles], weights)
    # plt.xlim([bounds_min[1], bounds_max[1]])
    # plt.show()

    # time.sleep(0.1)