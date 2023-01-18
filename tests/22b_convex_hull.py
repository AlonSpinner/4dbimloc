from bim4loc.geometry.minimal_distance import minimal_distance_from_projected_boundry
from bim4loc.geometry.convex_hull import convex_hull_jarvis as convex_hull
from bim4loc.geometry.utils import point_in_polygon
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from numpy.matlib import repmat
from bim4loc.binaries.paths import IFC_THREE_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import PcdSolid, ifc_converter, ScanSolid, weights2rgb
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors.sensors import Lidar
from bim4loc.sensors.models import inverse_lidar_model
from bim4loc.geometry.raycaster import NO_HIT
import bim4loc.existance_mapping.filters as filters
from copy import deepcopy
import time
import logging
import keyboard
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from bim4loc.geometry.pose2z import transform_from
import bim4loc.geometry.pose2z as pose2z

solids = [ifc_converter(IFC_PATH)[0]] #list of first element
world = RayCastingMap(solids)

drone = Drone(pose = np.array([-0.1, 1.0, 0.5, np.pi/6]))
sensor = Lidar(angles_u = np.linspace(-np.pi/4, np.pi/4, 50), 
               angles_v = np.linspace(-np.pi/4, np.pi/4, 50))
sensor.std = 0.05
sensor.piercing = False
sensor.max_range = 10.0
sensor.p0 = 0.4
drone.mount_sensor(sensor)
z, z_ids, z_normals, drone_p = drone.scan(world, n_hits = 1, 
                                                 project_scan = True, noisy = False)
#----------------- VISUALIZATION -----------------
#create world scene
visApp = VisApp()
[visApp.add_solid(s,"world") for s in world.solids]
visApp.redraw("world")
visApp.show_axes(False,"world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")
#pcd_scan and line_scan
pcd_scan = PcdSolid(shader = "normals")
visApp.add_solid(pcd_scan, "world")
line_scan = ScanSolid()
visApp.add_solid(line_scan,"world")
pcd_scan.update(drone_p.T, z_normals.reshape(-1,3))
line_scan.update(drone.pose[:3], drone_p.T)
visApp.redraw_all_scenes()
visApp.update_solid(pcd_scan)
visApp.update_solid(line_scan)

element_world_v = np.asarray(solids[0].geometry.vertices)
element_uv = pose2z.angle(drone.pose, element_world_v.T).T

hit_ray_uv = drone.sensor.uv[z_ids != NO_HIT]
dist_2_boundry = np.zeros(hit_ray_uv.shape[0])
for i, ray_uv in enumerate(hit_ray_uv):
    dist_2_boundry[i], _ = minimal_distance_from_projected_boundry(ray_uv, element_uv)
        

element_uv_hull = convex_hull(element_uv)
element_uv_hull_plus = np.vstack((element_uv_hull,element_uv_hull[0]))

fig, ax = plt.subplots()
ax.axis('equal')
ax.invert_xaxis()
ax.set_xlabel("yaw, rad"); ax.set_ylabel("pitch, rad")
ax.set_title("distance from boundry component")
ax.plot(element_uv_hull_plus[:,0],-element_uv_hull_plus[:,1]) #minus on the ys! for pitch convetion
sc = ax.scatter(hit_ray_uv[:,0],-hit_ray_uv[:,1],
                c=dist_2_boundry, s = 50)
fig.colorbar(sc)
plt.show()




