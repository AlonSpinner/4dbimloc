import numpy as np
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ArrowSolid, ifc_converter
from bim4loc.agents import Drone
import time

objects = ifc_converter(IFC_ONLY_WALLS_PATH)
visApp = VisApp()

for o in objects:
    visApp.add_solid(o)
visApp.redraw()
visApp.setup_default_camera("world")
visApp.show_axes()

drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
arrow = ArrowSolid(name = 'arrow', alpha = 1.0, pose =  drone.pose)
visApp.add_solid(drone.solid)
visApp.add_solid(arrow)

straight = np.array([0.5,0.0 ,0.0 ,0.0])
turn_left = np.array([0.0 ,0.0 ,0.0, np.pi/8])
turn_right = np.array([0.0, 0.0, 0.0, -np.pi/8])
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

time.sleep(0.5)
for a in actions:
    drone.move(a, 1e-9 * np.eye(4))
    arrow.update_geometry(drone.pose)

    visApp.update_solid(arrow)
    visApp.update_solid(drone.solid)

    time.sleep(0.05)
