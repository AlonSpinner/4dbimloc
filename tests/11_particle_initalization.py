import numpy as np
from bim4loc.geometry.poses import Pose2z
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors import Lidar
import time
import logging

np.random.seed(25)

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

current_time = 5.0 #[s]
solids = ifc_converter(IFC_PATH)

constructed_solids = []
for s in solids:
    s.set_random_completion_time()
    # if s.completion_time < current_time:
    constructed_solids.append(s.clone())
world = RayCastingMap(constructed_solids)

drone = Drone(pose = Pose2z(3,3,0, 1.5))
sensor = Lidar(angles_u = np.linspace(-np.pi/2, +np.pi/2, 36),
                 angles_v = np.linspace(-np.pi/30, +np.pi/30, 3)); 
sensor.std = 0.1
sensor.piercing = False
sensor.max_range = 100.0
drone.mount_sensor(sensor)

bounds_min, bounds_max, _ = world.bounds()
Nparticles = 100
particle_poses = []
for i in range(Nparticles):
    particle_poses.append(
        Pose2z(np.random.uniform(bounds_min[0], bounds_max[0]),
                np.random.uniform(bounds_min[1], bounds_max[1]),
                np.random.uniform(-np.pi, +np.pi),
                0)
    )

particles = ParticlesSolid(poses = particle_poses)

#create world scene
visApp = VisApp()
[visApp.add_solid(s,"world") for s in world.solids]
visApp.redraw("world")
visApp.show_axes(True,"world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")
visApp.add_solid(particles.lines, "world")
visApp.add_solid(particles.tails, "world")

u = Pose2z(1.0 ,0.0 ,0.0 ,0.0)
time.sleep(5)
for i in range(10):
    drone.move(u)

    particle_poses = [p.compose(u) for p in particle_poses]

    particles.update(particle_poses)

    visApp.update_solid(drone.solid)
    visApp.update_solid(particles.lines)
    visApp.update_solid(particles.tails)
    time.sleep(0.1)