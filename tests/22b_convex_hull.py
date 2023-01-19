from bim4loc.geometry.minimal_distance import minimal_distance_from_projected_boundry
from bim4loc.geometry.convex_hull import convex_hull_jarvis as convex_hull
import numpy as np
from bim4loc.binaries.paths import IFC_THREE_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import PcdSolid, ifc_converter
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors.sensors import Lidar
from bim4loc.geometry.raycaster import NO_HIT
import matplotlib.pyplot as plt
from bim4loc.geometry.pose2z import transform_from
import bim4loc.geometry.pose2z as pose2z

plt.rcParams['image.cmap'] = plt.cm.plasma
plt.rcParams['font.size'] = '24'

solids = [ifc_converter(IFC_PATH)[0]] #list of first element
world = RayCastingMap(solids)

drone = Drone(pose = np.array([2.0, 2.4, 1.5, 0 * np.pi/6]))
sensor = Lidar(angles_u = np.linspace(-np.pi/4, np.pi/4, 50), 
               angles_v = np.linspace(-np.pi/4, np.pi/4, 50))
sensor.piercing = False
sensor.max_range = 10.0
drone.mount_sensor(sensor)

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

nrm_hit_ray_uv = hit_ray_uv/np.array([np.pi, np.pi/2])
nrm_element_uv = element_uv/np.array([np.pi, np.pi/2])

for i, ray in enumerate(nrm_hit_ray_uv):
    weight_dist_2_boundry[i], _ = minimal_distance_from_projected_boundry(ray, nrm_element_uv)
        

nrm_element_uv_hull = convex_hull(nrm_element_uv)
nrm_element_uv_hull_plus = np.vstack((nrm_element_uv_hull,nrm_element_uv_hull[0]))

fig = plt.figure(figsize = (7,8))
ax = fig.add_subplot(111)
ax.invert_xaxis()
ax.set_xlabel(r"yaw / $\pi$"); ax.set_ylabel(r"pitch / $\frac{1}{2}\pi$")
# ax.set_title("distance from boundry component")
ax.plot(nrm_element_uv_hull_plus[:,0], nrm_element_uv_hull_plus[:,1], c = "k", lw = 3)
sc = ax.scatter(nrm_hit_ray_uv[:,0], nrm_hit_ray_uv[:,1],
                c=weight_dist_2_boundry, s = 50)
cbar = fig.colorbar(sc)
ax.grid(True)
ax.set_xlim(-0.33,0.33)
ax.set_aspect(1.0)
plt.show()

#-----------------------cosine componenet-----------------------
weight_cos_incident = np.abs(z_cos_incident[z_ids != NO_HIT])

fig = plt.figure(figsize = (7,8))
ax = fig.add_subplot(111)
ax.axis('equal')
ax.invert_xaxis()
ax.set_xlabel(r"yaw / $\pi$"); ax.set_ylabel(r"pitch / $\frac{1}{2}\pi$")
ax.set_title("cos incident component")
ax.plot(nrm_element_uv_hull_plus[:,0], nrm_element_uv_hull_plus[:,1], c = "k", lw = 3)
sc = ax.scatter(nrm_hit_ray_uv[:,0], nrm_hit_ray_uv[:,1],
                c=weight_cos_incident, s = 50)
fig.colorbar(sc)
ax.grid(True)
ax.set_xlim(-0.33,0.33)
ax.set_aspect(1.0)
plt.show()

#-----------------------combined-----------------------
fig = plt.figure(figsize = (7,8))
ax = fig.add_subplot(111)
ax.invert_xaxis()
ax.set_xlabel(r"yaw / $\pi$"); ax.set_ylabel(r"pitch / $\frac{1}{2}\pi$")
# ax.set_title("full weight")
ax.plot(nrm_element_uv_hull_plus[:,0], nrm_element_uv_hull_plus[:,1], c = "k", lw = 3)
sc = ax.scatter(nrm_hit_ray_uv[:,0], nrm_hit_ray_uv[:,1],
                c=weight_cos_incident * weight_dist_2_boundry, s = 50)
cbar = fig.colorbar(sc)
ax.grid(True)
ax.set_xlim(-0.33,0.33)
ax.set_aspect(1.0)
plt.show()





