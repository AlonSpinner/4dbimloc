import numpy as np
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ScanSolid, ParticlesSolid
from bim4loc.agents import Drone
from bim4loc.sensors.sensors import Lidar
from bim4loc.maps import RayCastingMap
from bim4loc.geometry.pose2z import compose_s
from bim4loc.random.one_dim import Gaussian
import time
import logging
import copy
import keyboard
from bim4loc.sensors.models import inverse_lidar_model

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

#FUNCTIONS
gaussian_pdf = Gaussian._pdf

#BUILD WORLD
solids = ifc_converter(IFC_ONLY_WALLS_PATH)
world = RayCastingMap(solids)
bounds_min, bounds_max, extent = world.bounds()
beliefs = np.ones(len(world.solids),dtype = np.float64)

#INITALIZE DRONE AND SENSOR
drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar(angles_u = np.linspace(-np.pi/2,np.pi/2,36), angles_v = np.array([0.0])); 
sensor.std = 0.1; sensor.piercing = False; sensor.max_range = 100.0
drone.mount_sensor(sensor)

simulated_sensor = copy.deepcopy(sensor)
simulated_sensor.std =  5.0 * sensor.std

straight = np.array([0.5,0.0 ,0.0 ,0.0])
turn_left = np.array([0.0 ,0.0 ,0.0, np.pi/8])
turn_right = np.array([0.0, 0.0, 0.0, -np.pi/8])
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

#SPREAD PARTICLES
N_circum_particles = 30
N_radii_particles = 10
N_particles = N_circum_particles * N_radii_particles
particles = np.tile(drone.pose, (N_particles,1))
theta = np.linspace(-np.pi, np.pi, N_circum_particles)
r = np.linspace(0.1, 1.0, N_radii_particles)
rr, tt = np.meshgrid(r, theta)
particles[:,0] = drone.pose[0] + rr.flatten() * np.cos(tt.flatten())
particles[:,1] = drone.pose[1] + rr.flatten() * np.sin(tt.flatten())
particles[:,2] = drone.pose[2] -1.0


#INITALIZE WEIGHTS
weights1 = np.ones(N_particles) / N_particles
weights2 = weights1.copy()

visApp = VisApp()
for s in solids:
    visApp.add_solid(s)
visApp.redraw()
visApp.show_axes(False)
visApp.setup_default_camera()
visApp.add_solid(drone.solid)
vis_particles1 = ParticlesSolid(poses = particles)
visApp.add_solid(vis_particles1.lines)
visApp.add_solid(vis_particles1.tails)
vis_scan = ScanSolid("scan")
visApp.add_solid(vis_scan)
visApp.redraw()

visApp.add_scene("world2", "world")
for s in solids:
    visApp.add_solid(s, "world2")
visApp.redraw("world2")
visApp.show_axes(False, "world2")
visApp.setup_default_camera("world2")
visApp.add_solid(drone.solid, "world2")
vis_particles2 = ParticlesSolid(poses = particles)
visApp.add_solid(vis_particles2.lines, "world2")
visApp.add_solid(vis_particles2.tails, "world2")
vis_scan = ScanSolid("scan", "world2")
visApp.add_solid(vis_scan, "world2")
visApp.redraw("world2")

U_COV = np.diag([0.05, 0.05, 0.0, np.radians(1.0)])
#LOOP
time.sleep(0.1)
# keyboard.wait('space')
for t, u in enumerate(actions):
    #move drone
    drone.move(u)
    
    #produce measurement
    z, z_ids, z_normals, z_p = drone.scan(world, project_scan = True, noisy = False)

    #---------------------------FILTER-------------------------------------
    #compute weights and normalize
    sum_weights1 = 0.0
    sum_weights2 = 0.0
    noisy_u = np.random.multivariate_normal(u, U_COV, N_particles)
    for i in range(N_particles):
        particles[i] = compose_s(particles[i], u)
        # particles[i] = compose_s(particles[i], noisy_u[i])


        particle_z_values, particle_z_ids, _, particle_z_cos_incident,_ \
            = simulated_sensor.sense(particles[i], 
                                     world, n_hits = 5, 
                                     noisy = False)
        
        # particle_stds = simulated_sensor.std#/np.maximum(np.abs(particle_z_cos_incident), 1e-16)
        # pz = 0.9 + 0.1 * gaussian_pdf(particle_z_values, particle_stds, z, pseudo = True)        
        #line 205 in https://github.com/ros-planning/navigation/blob/noetic-devel/amcl/src/amcl/sensors/amcl_laser.cpp
        
        pz = np.zeros(len(z))
        for j in range(len(z)):
            _, pz[j] = inverse_lidar_model(z[j], 
                                        np.array([particle_z_values[j]]),
                                        np.array([particle_z_ids[j]]), 
                                        beliefs, 
                             simulated_sensor.std,simulated_sensor.max_range)

        weights1[i] = np.product(pz)
        weights2[i] = 1.0 + np.sum(np.power(pz,3))
        
        sum_weights1 += weights1[i]
        sum_weights2 += weights2[i]

    #normalize
    weights1 = weights1 / sum_weights1
    weights2 = weights2 / sum_weights2

    vis_particles1.update(particles, weights1)
    visApp.update_solid(vis_particles1.lines)
    visApp.update_solid(vis_particles1.tails)
    vis_scan.update(drone.pose[:3], z_p.T)
    visApp.update_solid(drone.solid)
    visApp.update_solid(vis_scan)
    visApp.redraw()

    vis_particles2.update(particles, weights2)
    visApp.update_solid(vis_particles2.lines, "world2")
    visApp.update_solid(vis_particles2.tails, "world2")
    vis_scan.update(drone.pose[:3], z_p.T)
    visApp.update_solid(drone.solid, "world2")
    visApp.update_solid(vis_scan, "world2")
    visApp.redraw("world2")

    time.sleep(1)
    keyboard.wait('space')

    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    def crop_image(image, crop_ratio_w, crop_ratio_h):
        h,w = image.shape[:2]
        crop_h = int(h * crop_ratio_h/2)
        crop_w = int(w * crop_ratio_w/2)
        return image[crop_h:-crop_h, crop_w:-crop_w,:]
    images_output_path = os.path.join(dir_path, "04c2_images")
    images = visApp.get_images(images_output_path,prefix = f"{t}_")
    

print('finished')
