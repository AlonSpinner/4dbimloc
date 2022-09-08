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

import open3d as o3d

solids = ifc_converter(IFC_ONLY_WALLS_PATH)
world = RayCastingMap(solids)

min_bounds, max_bounds, _, = world.bounds()
mid = (min_bounds + max_bounds) / 2

drone = Drone(pose = Pose2z(mid[0],mid[1],0,mid[2]))
sensor = Lidar(angles_u = np.linspace(-np.pi,np.pi,4), angles_v = np.array([0.0])); 
sensor.std = 0.05; sensor.piercing = False
sensor.max_range = 1000.0
drone.mount_sensor(sensor)

visApp = VisApp()
for o in solids:
    visApp.add_solid(o,"world")
visApp.redraw()
visApp.setup_default_camera("world")
visApp.show_axes()

visApp.add_solid(drone.solid)
pcd_scan = PcdSolid(shader = "normals")
visApp.add_solid(pcd_scan)
time.sleep(1)

s = time.time()
N = 100
for i in range(N):
    # keyboard.wait('space')
    z, z_ids, z_normals, p = drone.scan(world, project_scan = True)
e = time.time()

print(f"Time per scan: {(e-s)/N:.6f} s")
print(f"Time per ray: {(e-s)/N/(sensor._Nu * sensor._Nv):.6f} s")
