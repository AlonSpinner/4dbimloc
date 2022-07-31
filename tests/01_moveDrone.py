import numpy as np
import open3d as o3d
from bim4loc.geometry import pose2
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, DRONE_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solid_objects import ifc_converter, DynamicObject
import time

objects = ifc_converter(IFC_ONLY_WALLS_PATH)
visApp = VisApp()

for o in objects:
    visApp.add_object(o)
visApp.reset_camera_to_default()

base_drone_geo = o3d.io.read_triangle_mesh(DRONE_PATH)
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"
mat.base_color = [1.0 , 0.0 , 0.0 , 1.0]
x = pose2(3,3,0)
drone = DynamicObject(name = 'drone', geometry = base_drone_geo, material = mat, pose = x)
visApp.add_object(drone)

time.sleep(3)
actions = [pose2(1,0,np.pi/10)] * 10

for a in actions:
    x = x + a
    drone.update_geometry(x, z = 1.5)
    visApp.update_object(drone)

    time.sleep(0.2)
