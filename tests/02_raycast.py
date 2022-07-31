import numpy as np
import open3d as o3d
from bim4loc.geometry import pose2
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, DRONE_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solid_objects import ifc_converter, PcdObject
from bim4loc.agents import Drone
from bim4loc.maps import Map
import time

objects = ifc_converter(IFC_ONLY_WALLS_PATH)
drone = Drone(pose2 = pose2(3,3,0), hover_height = 1.5)
world = Map(objects)
pcd_scan = PcdObject()

straight = pose2(0.5,0,0)
turn_left = pose2(0,0,np.pi/8)
turn_right = pose2(0,0,-np.pi/8)
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

visApp = VisApp()
for o in objects:
    visApp.add_object(o)
visApp.reset_camera_to_default()
visApp.add_object(drone.object)
time.sleep(1)

for a in actions:
    drone.move(a)
    z, p = drone.scan(world)
    pcd_scan.update(p)

    visApp.update_object(drone.object)
    visApp.update_object(pcd_scan)

    time.sleep(0.1)
