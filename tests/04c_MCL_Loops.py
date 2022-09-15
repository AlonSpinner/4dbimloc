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
import keyboard

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

#FUNCTIONS
gaussian_pdf = Gaussian._pdf

#BUILD WORLD
solids = ifc_converter(IFC_ONLY_WALLS_PATH)
world = RayCastingMap(solids)

#INITALIZE DRONE AND SENSOR
drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar(angles_u = np.linspace(-np.pi/2,np.pi/2,2), angles_v = np.array([0.0])); 
sensor.std = 0.1; sensor.piercing = False; sensor.max_range = 100.0
drone.mount_sensor(sensor)

simulated_sensor = copy.deepcopy(sensor)
simulated_sensor.std =  5.0 * sensor.std

straight = np.array([0.5,0.0 ,0.0 ,0.0])
turn_left = np.array([0.0 ,0.0 ,0.0, np.pi/8])
turn_right = np.array([0.0, 0.0, 0.0, -np.pi/8])
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

#SPREAD PARTICLES UNIFORMLY
bounds_min, bounds_max, extent = world.bounds()
N_circum_particles = 30
N_radii_particles = 10
N_particles = N_circum_particles * N_radii_particles

#debugging
particles = np.tile(drone.pose, (N_particles,1))
theta = np.linspace(-np.pi, np.pi, N_circum_particles)
r = np.linspace(0.1, 1.0, N_radii_particles)
rr, tt = np.meshgrid(r, theta)
particles[:,0] = drone.pose[0] + rr.flatten() * np.cos(tt.flatten())
particles[:,1] = drone.pose[1] + rr.flatten() * np.sin(tt.flatten())


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
ALPHA_SLOW = 0.1 #0.0 <= ALPHA_SLOW << ALPHA_FAST, also: http://wiki.ros.org/amcl
ALPHA_FAST = 0.3
POSE_MIN_BOUNDS = np.array([bounds_min[0],bounds_min[1], 0.0 , -np.pi])
POSE_MAX_BOUNDS = np.array([bounds_max[0],bounds_max[1], 0.0 , np.pi])
w_slow = w_fast = 0.0
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
    sum_weights = 0.0
    noisy_u = np.random.multivariate_normal(u, U_COV, N_particles)
    for i in range(N_particles):
        particles[i] = compose_s(particles[i], u)
        # particles[i] = compose_s(particles[i], noisy_u[i])

        if np.any(particles[i][:3] < bounds_min[:3]) \
             or np.any(particles[i][:3] > bounds_max[:3]):
            weights[i] = 0.0
            continue

        particle_z_values, particle_z_ids, _ = simulated_sensor.sense(particles[i], 
                                                            world, n_hits = 10, 
                                                            noisy = False)
        
        pz = 0.4 + 0.6 * gaussian_pdf(particle_z_values, simulated_sensor.std, z, pseudo = True)
        
        #line 205 in https://github.com/ros-planning/navigation/blob/noetic-devel/amcl/src/amcl/sensors/amcl_laser.cpp
        # weights[i] = 1.0 + np.sum(pz**3)
        weights[i] = np.product(pz)
        
        sum_weights += weights[i]
    
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
    if 0: #n_eff < ETA_THRESHOLD or (t % 2) == 0:
        new_particles = np.zeros_like(particles)
        
        c = np.cumsum(weights)
        
        w_diff = max(1.0 - w_fast / w_slow, 0.0)

        print(f"resampling with w_diff = {w_diff}")
        i = 0
        while i < N_particles:
            if np.random.uniform() < w_diff:
                new_particles[i] = np.random.uniform(POSE_MIN_BOUNDS, POSE_MAX_BOUNDS)
                i += 1
            else:
                r = np.random.uniform()
                for j in range(N_particles):
                    if c[j] <= r and r < c[j+1]:
                        break

                if weights[j] > 0.0:
                    new_particles[i] = particles[j]
                    i += 1
        
        particles = new_particles
        weights = np.ones(N_particles) / N_particles

        #Reset averages, to avoid spiraling off into complete randomness.
        if w_diff > 0.0:
            w_slow = w_fast = 0.0

    vis_particles.update(particles, weights)
    visApp.update_solid(vis_particles.lines)
    visApp.update_solid(vis_particles.tails)
    vis_scan.update(drone.pose[:3], z_p.T)
    visApp.update_solid(drone.solid)
    visApp.update_solid(vis_scan)
    visApp.redraw()

    # time.sleep(0.01)
    keyboard.wait('space')
    

print('finished')
