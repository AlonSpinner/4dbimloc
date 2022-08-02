import numpy as np
from bim4loc.geometry import pose2z
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import PcdSolid, ifc_converter, PcdSolid, Arrow
from bim4loc.agents import Drone
from bim4loc.maps import Map
from bim4loc.filters import vanila_SE2
import time
import logging

logging.basicConfig(format = '%(levelname)s in %(funcName)s: %(message)s')

solids = ifc_converter(IFC_ONLY_WALLS_PATH)
drone = Drone(pose = pose2z(3,3,0, 1.5))
world = Map(solids)

straight = pose2z(0.5,0,0,0)
turn_left = pose2z(0,0,np.pi/8,0)
turn_right = pose2z(0,0,-np.pi/8,0)
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

model = world
min_bounds, max_bounds = model.bounds()

Nparticles = 100
inital_poses = []
for i in range(Nparticles):
    inital_poses.append(
        pose2z(np.random.uniform(min_bounds[0], max_bounds[0]),
            np.random.uniform(min_bounds[1], max_bounds[1]),
            np.random.uniform(min_bounds[2], max_bounds[2]),
            drone.pose._z)
    )
arrows = []
for i in range(Nparticles):
    arrows.append(Arrow(name = f'arrow_{i}', alpha = 1/Nparticles, pose = inital_poses[i]))
pf = vanila_SE2(drone, model , inital_poses)

visApp = VisApp()
for s in solids:
    visApp.add_solid(s)
visApp.show_axes(True)
visApp.reset_camera_to_default()
for a in arrows:
    visApp.add_solid(a)
visApp.add_solid(drone.solid)
pcd_scan = PcdSolid()
visApp.add_solid(pcd_scan)

time.sleep(1)
for a in actions:
    drone.move(a, 1e-9 * np.eye(4))
    z, p = drone.scan(world, std = 0.1)
    
    pcd_scan.update(p)
    visApp.update_solid(drone.solid)
    visApp.update_solid(pcd_scan)

    time.sleep(0.1)

print('finished')
