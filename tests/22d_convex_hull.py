import numpy as np
problem_pose = np.array([9.49095584e+00, 7.00604226e+00, 2.00000000e+00, 8.04164683e-03])
problem_elem = 11

import numpy as np
from bim4loc.binaries.paths import IFC_ARENA_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, PcdSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors.sensors import Lidar
from bim4loc.geometry.raycaster import NO_HIT
from copy import deepcopy
from bim4loc.geometry.minimal_distance import minimal_distance_from_projected_boundry
from bim4loc.geometry.convex_hull import convex_hull_jarvis as convex_hull
import matplotlib.pyplot as plt
import bim4loc.geometry.pose2z as pose2z
from bim4loc.geometry.pose2z import transform_from
plt.rcParams['image.cmap'] = plt.cm.plasma

solids = ifc_converter(IFC_PATH)
solids = [solids[11]]
world = RayCastingMap(solids)



#INITALIZE DRONE AND SENSOR
drone = Drone(pose = problem_pose)
sensor = Lidar(angles_u = np.linspace(-np.pi-0.2,-np.pi+0.05, 3), angles_v = np.array([0.0])); 
sensor.std = 0.05; sensor.piercing = False; sensor.max_range = 10.0
drone.mount_sensor(sensor)

simulated_sensor = deepcopy(sensor)
simulated_sensor.piercing = True
simulated_sensor.p0 = 0.1
simulated_sensor.max_range_cutoff = False

z, z_ids, z_normals, z_cos_incident, z_n_hits = sensor.sense(drone.pose, world, noisy = False)

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
sensor_p = sensor.scan_to_points(z)
drone_p = transform_from(drone.pose, sensor_p)
pcd_scan.update(drone_p.T, z_normals.reshape(-1,3))
visApp.redraw_all_scenes()
visApp.update_solid(pcd_scan)

hit_ray_uv = drone.sensor.uv[z_ids != NO_HIT]
#-----------------------distance from boundry component-----------------------
element_world_v = np.asarray(solids[0].geometry.vertices)
element_uv = pose2z.angle(drone.pose, element_world_v.T).T
weight_dist_2_boundry = np.zeros(hit_ray_uv.shape[0])

for i, ray in enumerate(hit_ray_uv):
    weight_dist_2_boundry[i], _ = minimal_distance_from_projected_boundry(ray, element_uv)
print(weight_dist_2_boundry)
        