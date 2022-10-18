import numpy as np
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ScanSolid
from bim4loc.agents import Drone
from bim4loc.sensors.sensors import Lidar
from bim4loc.maps import RayCastingMap
import time
import keyboard

import open3d as o3d

solids = ifc_converter(IFC_ONLY_WALLS_PATH)
drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar(angles_u = np.linspace(-np.pi/3, np.pi/3, 5), angles_v = np.array([0])); 
sensor.std = 0.05
sensor.max_range = 1000.0
drone.mount_sensor(sensor)
world = RayCastingMap(solids)

straight = np.array([0.5,0.0 ,0.0 ,0.0])
turn_left = np.array([0.0 ,0.0 ,0.0, np.pi/8])
turn_right = np.array([0.0, 0.0, 0.0, -np.pi/8])
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

visApp = VisApp()
for o in solids:
    visApp.add_solid(o,"world")
visApp.redraw()
visApp.setup_default_camera("world")
visApp.show_axes()

vis_scan = ScanSolid("scan")
visApp.add_solid(vis_scan)
visApp.redraw()
visApp.add_solid(drone.solid)
time.sleep(1)

for a in actions:
    # keyboard.wait('space')
    drone.move(a)
    z, z_ids, z_normals, p = drone.scan(world, project_scan = True)
    for s in world.solids:
        if s in z_ids:
            s.material.base_color = (1,0,0,1)
        else:
            s.material.base_color = np.hstack((s.ifc_color,1))

    vis_scan.update(drone.pose[:3], p.T)
    [visApp.update_solid(s) for s in world.solids]
    visApp.update_solid(drone.solid)
    visApp.update_solid(vis_scan)
