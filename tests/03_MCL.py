import numpy as np
from bim4loc.geometry import pose2
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solid_objects import ifc_converter, PcdObject, Arrow
from bim4loc.agents import Drone
from bim4loc.maps import Map
from bim4loc.filters import vanila_SE2
import time

# import open3d as o3d
# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

objects = ifc_converter(IFC_ONLY_WALLS_PATH)
drone = Drone(pose2 = pose2(3,3,0), hover_height = 1.5)
world = Map(objects)

straight = pose2(0.5,0,0)
turn_left = pose2(0,0,np.pi/8)
turn_right = pose2(0,0,-np.pi/8)
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

model = world
min_bounds, max_bounds = model.bounds()

Nparticles = 100
inital_poses = []
for i in range(Nparticles):
    inital_poses.append(
        pose2(np.random.uniform(min_bounds[0], max_bounds[0]),
            np.random.uniform(min_bounds[1], max_bounds[1]),
            np.random.uniform(min_bounds[2], max_bounds[2]))
    )
arrows = []
for i in range(Nparticles):
    arrows.append(Arrow(name = f'arrow_{i}', alpha = 1/Nparticles, pose = inital_poses[i]))
pf = vanila_SE2(drone, model , inital_poses)

visApp = VisApp()
for o in objects:
    visApp.add_object(o)
visApp.show_axes(True)
visApp.reset_camera_to_default()
for a in arrows:
    visApp.add_object(a)
visApp.add_object(drone.object)
pcd_scan = PcdObject()
visApp.add_object(pcd_scan)

time.sleep(1)
for a in actions:
    drone.move(a, 1e-9 * np.eye(3))
    z, p = drone.scan(world, std = 0.1)
    
    
    pcd_scan.update(p)
    visApp.update_object(drone.object)
    visApp.update_object(pcd_scan)

    time.sleep(0.2)

print('finished')
