import numpy as np
from bim4loc.geometry.poses import Pose2z
from bim4loc.binaries.paths import IFC_LINE_UP_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid, ScanSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors import Lidar
import time
import logging

np.random.seed(25)
logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

#BUILD WORLD
solids = ifc_converter(IFC_PATH)
world = RayCastingMap(solids)

#INITALIZE DRONE AND SENSOR
drone = Drone(pose = Pose2z(3.0 ,1.0 ,0.0 , 1.5))
sensor = Lidar(angles_u = np.array([0.0]),
                 angles_v = np.array([0.0])); 
sensor.std = 0.1
sensor.piercing = False
sensor.max_range = 100.0
drone.mount_sensor(sensor)
U_COV = np.diag([0.1, 0.0, 0.0, 0.0])

#SPREAD PARTICLES UNIFORMLY
bounds_min, bounds_max, _ = world.bounds()
Nparticles = 100
particle_poses = []
for i in range(Nparticles):
    particle_poses.append(
        Pose2z(3.0,
                np.random.uniform(bounds_min[1], bounds_max[1]),
                0.0,
                0.0)
    )
vis_particles = ParticlesSolid(poses = particle_poses)

#DRAW
visApp = VisApp()
[visApp.add_solid(s,"world") for s in world.solids]
visApp.redraw("world")
visApp.show_axes(True,"world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")
visApp.add_solid(vis_particles.lines, "world")
visApp.add_solid(vis_particles.tails, "world")
vis_scan = ScanSolid("scan")
visApp.add_solid(vis_scan, "world")

#LOOP
u = Pose2z(0.0 ,1.0 ,0.0 ,0.0)
for i in range(10):
    #move drone
    drone.move(u, U_COV)
    
    #produce measurement
    z, z_ids, z_normals, z_p = drone.scan(world, project_scan = True)

    particle_poses = [p.compose(u) for p in particle_poses]

    #updating drawings
    vis_scan.update(drone.pose.t, z_p)
    vis_particles.update(particle_poses)
    visApp.update_solid(vis_scan)
    visApp.update_solid(drone.solid)
    visApp.update_solid(vis_particles.lines)
    visApp.update_solid(vis_particles.tails)
    time.sleep(0.3)