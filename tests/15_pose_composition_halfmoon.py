import numpy as np
from bim4loc.agents import Drone
from bim4loc.visualizer import VisApp
from bim4loc.solids import ParticlesSolid
from bim4loc.geometry.pose2z import compose_s
from bim4loc.random.multi_dim import sample_normal
import time
import os 

drone = Drone(pose = np.array([0.0, 0.0, 0.5, 0.0]))

U_COV = np.diag([0.1, 0.1, 0.0, np.radians(5.0)])

particles = np.tile(drone.pose, (100,1))

u = np.array([1,0.0 ,0.0 ,0.0])

visApp = VisApp()
visApp.add_solid(drone.solid)
vis_particles = ParticlesSolid(poses = particles)
visApp.add_solid(vis_particles.lines)
visApp.add_solid(vis_particles.tails)
visApp.setup_default_camera()
visApp.show_axes()
visApp.redraw()

for t in range(10):
    drone.move(u)

    # noisy_u = np.random.multivariate_normal(u, U_COV, len(particles))
    noisy_u = sample_normal(u, U_COV, len(particles))
    for i in range(len(particles)):
        particles[i] = compose_s(particles[i], noisy_u[i]) 

    visApp.update_solid(drone.solid)
    vis_particles.update(particles)
    visApp.update_solid(vis_particles.lines)
    visApp.update_solid(vis_particles.tails)
    visApp.redraw()

    visApp.get_images(dir_path =  os.path.dirname(os.path.realpath(__file__)), sufix = f"{t}")
    time.sleep(0.5)


