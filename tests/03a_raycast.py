import numpy as np
from bim4loc.geometry.poses import Pose2z
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, PcdSolid
from bim4loc.agents import Drone
from bim4loc.sensors import Lidar
from bim4loc.maps import RayCastingMap
import time
import keyboard

solids = ifc_converter(IFC_ONLY_WALLS_PATH)
drone = Drone(pose = Pose2z(3,3,0,1.5))
sensor = Lidar(); sensor.std = 0.05; sensor.piercing = False
sensor.max_range = 1000.0
drone.mount_sensor(sensor)
world = RayCastingMap(solids)

straight = Pose2z(0.5,0,0,0)
turn_left = Pose2z(0,0,np.pi/8,0)
turn_right = Pose2z(0,0,-np.pi/8,0)
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

visApp = VisApp()
for o in solids:
    visApp.add_solid(o,"world")
visApp.redraw()
visApp.setup_default_camera("world")
visApp.show_axes()

visApp.add_solid(drone.solid)
pcd_scan = PcdSolid()
visApp.add_solid(pcd_scan)

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

    pcd_scan.update(p.T)

    [visApp.update_solid(s) for s in world.solids]
    visApp.update_solid(drone.solid)
    visApp.update_solid(pcd_scan)

    # time.sleep(1)
