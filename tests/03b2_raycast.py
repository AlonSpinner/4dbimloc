import numpy as np
from bim4loc.geometry.pose2z import transform_from
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, PcdSolid, LinesSolid, ScanSolid
from bim4loc.agents import Drone
from bim4loc.sensors.sensors import Lidar
from bim4loc.maps import RayCastingMap
from bim4loc.geometry.raycaster import NO_HIT
import time
import keyboard
from numba import njit, prange
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

#https://numba.pydata.org/numba-doc/0.37.0/user/troubleshoot.html

solids = ifc_converter(IFC_ONLY_WALLS_PATH)
drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar()
sensor.std = 0.05; sensor.max_range = 100.0
drone.mount_sensor(sensor)
world = RayCastingMap(solids)

straight = np.array([0.5,0.0 ,0.0 ,0.0])
turn_left = np.array([0.0 ,0.0 ,0.0, np.pi/8])
turn_right = np.array([0.0, 0.0, 0.0, -np.pi/8])
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

visApp = VisApp()
for s in solids:
    visApp.add_solid(s,"world")
visApp.redraw()
visApp.setup_default_camera("world")
visApp.show_axes()

visApp.add_solid(drone.solid)
pcd_scan = PcdSolid(shader = "normals")
# pcd_scan.material.point_size = 5.0
visApp.add_solid(pcd_scan)
line_scan = ScanSolid()
visApp.add_solid(line_scan)


sense_fcn = sensor.get_sense_piercing(world, n_hits = 10, noisy = False)

# @njit(parallel = True, debug= True)
def sense_fcn_1000(pose):
    for i in prange(1000):
        sense_fcn(pose)
    

time.sleep(1)
for a in actions:
    # keyboard.wait('space')
    
    drone.move(a)
    z, z_ids, z_normals, _, _ = sense_fcn(drone.pose)
    # sense_fcn_1000(drone.pose)
    
    drone_p = sensor.scan_to_points(z)
    p = transform_from(drone.pose,drone_p)

    z_ids_flat = z_ids.flatten()
    for s_i in range(len(world.solids)):
        if s_i in z_ids_flat:
            world.solids[s_i].material.base_color = (1,0,0,1)
        else:
            world.solids[s_i].material.base_color = np.hstack((s.ifc_color,1))

    pcd_scan.update(p.T, z_normals.reshape(-1,3))
    line_scan.update(drone.pose[:3], p.T)
    [visApp.update_solid(s) for s in world.solids]
    visApp.update_solid(drone.solid)
    visApp.update_solid(pcd_scan)
    visApp.update_solid(line_scan)