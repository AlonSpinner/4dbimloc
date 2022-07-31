import numpy as np
import open3d as o3d
from bim4loc.geometry import pose2
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, DRONE_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solid_objects import ifc_converter, o3dObject
import time

objects = ifc_converter(IFC_ONLY_WALLS_PATH)
visApp = VisApp()

for o in objects:
    visApp.add_object(o)

visApp.reset_camera_to_default()
time.sleep(1)


drone_geo = o3d.io.read_triangle_mesh(DRONE_PATH)
x = pose2(3,3,0)
drone_geo.transform(x.T3d(z = 1.5))
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"
mat.base_color = [1.0 , 0 , 0 , 1.0]

drone = o3dObject(name = 'drone', geometry = drone_geo, material = mat)
visApp.add_object(drone)

actions = [pose2(1,0,np.pi/10)] * 10
for a in actions:
    drone.geometry.rotate(a.R3d(),x.t3d())
    drone.geometry.translate(a.t3d())
    visApp.update_object(drone)

    x = x + a
    print(x)
    time.sleep(0.2)

# time.sleep(50.0)
# vis.destroy_window()
