import numpy as np
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ScanSolid, ParticlesSolid
from bim4loc.agents import Drone
from bim4loc.sensors import Lidar
from bim4loc.maps import RayCastingMap
from bim4loc.geometry.pose2z import compose_s
from bim4loc.random.one_dim import Gaussian
import time
import logging
import copy

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

#FUNCTIONS
gaussian_pdf = Gaussian._pdf

#BUILD WORLD
solids = ifc_converter(IFC_ONLY_WALLS_PATH)
world = RayCastingMap(solids)

#INITALIZE DRONE AND SENSOR
drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar(angles_u = np.linspace(-np.pi/2,np.pi/2,36), angles_v = np.array([0.0])); 
sensor.std = 0.1; sensor.piercing = False; sensor.max_range = 100.0
drone.mount_sensor(sensor)

straight = np.array([0.5,0.0 ,0.0 ,0.0])
turn_left = np.array([0.0 ,0.0 ,0.0, np.pi/8])
turn_right = np.array([0.0, 0.0, 0.0, -np.pi/8])
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

#SPREAD PARTICLES UNIFORMLY
bounds_min, bounds_max, extent = world.bounds()
N_particles = 100
particles = np.vstack((np.random.uniform(bounds_min[0], bounds_max[0], N_particles),
                       np.random.uniform(bounds_min[1], bounds_max[1], N_particles),
                       np.zeros(N_particles),
                       np.random.uniform(-np.pi, np.pi, N_particles))).T
# particles[0] = drone.pose #<------------------------ CHEATTTTINGGG !!
#INITALIZE WEIGHTS
weights = np.ones(N_particles) / N_particles

visApp = VisApp()
for s in solids:
    visApp.add_solid(s)
visApp.redraw()
visApp.show_axes()
visApp.setup_default_camera()
visApp.add_solid(drone.solid)
vis_particles = ParticlesSolid(poses = particles)
visApp.add_solid(vis_particles.lines)
visApp.add_solid(vis_particles.tails)
vis_scan = ScanSolid("scan")
visApp.add_solid(vis_scan)
visApp.redraw()

U_COV = np.diag([0.05, 0.05, 0.0, np.radians(1.0)])
ETA_THRESHOLD = 1.0/N_particles
#LOOP
time.sleep(0.1)
for t, u in enumerate(actions):
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

        if np.any(particles[i][:3] < bounds_min[:3]) \
             or np.any(particles[i][:3] > bounds_max[:3]):
            weights[i] = 0.0
            continue

        particle_z_values, particle_z_ids, _ = sensor.sense(particles[i], 
                                                            world, n_hits = 10, 
                                                            noisy = False)
        
        pz = 0.3 + 0.7 * gaussian_pdf(particle_z_values, sensor.std, z, pseudo = True)
        
        #line 229 in https://github.com/atinfinity/amcl/blob/master/src/amcl/sensors/amcl_laser.cpp
        weights[i] *= (1.0 + np.sum(pz**3))

        # weights[i] *= np.product(pz)
        sum_weights += weights[i]
    #normalize
    weights = weights / sum_weights
    
    #resample
    n_eff = weights.dot(weights)
    if n_eff < ETA_THRESHOLD or (t % 10) == 0:
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

    vis_particles.update(particles)
    visApp.update_solid(vis_particles.lines)
    visApp.update_solid(vis_particles.tails)
    vis_scan.update(drone.pose[:3], z_p.T)
    visApp.update_solid(drone.solid)
    visApp.update_solid(vis_scan)

    time.sleep(0.01)

print('finished')
