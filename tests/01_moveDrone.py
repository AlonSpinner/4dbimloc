import numpy as np
import open3d as o3d
from bim4loc.geometry import pose2
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, DRONE_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solid_objects import ifc_converter
from bim4loc.agents import Drone
import time

objects = ifc_converter(IFC_ONLY_WALLS_PATH)
visApp = VisApp()
time.sleep(1)

for o in objects:
    visApp.add_object(o)
visApp.reset_camera_to_default()

drone = Drone(pose2 = pose2(3,3,0), z = 1.5)
visApp.add_object(drone.object)

straight = pose2(0.5,0,0)
turn_left = pose2(0,0,np.pi/8)
turn_right = pose2(0,0,-np.pi/8)
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

for a in actions:
    drone.move(a)
    visApp.update_object(drone.object)

    time.sleep(0.1)
