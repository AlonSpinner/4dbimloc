import numpy as np
from numpy.matlib import repmat
from bim4loc.geometry.poses import Pose2z
from bim4loc.binaries.paths import IFC_THREE_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import IfcSolid, PcdSolid, LinesSolid, ifc_converter
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors import Lidar1D
from bim4loc.geometry.raycaster import NO_HIT
import bim4loc.existance_mapping.filters as filters
from copy import deepcopy
import time
import logging
import keyboard
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

np.set_printoptions(precision=3)

solids = ifc_converter(IFC_PATH)
world = RayCastingMap(solids)

drone = Drone(pose = Pose2z(3.0,2.5, 0, 1.5))
sensor = Lidar1D(); sensor.std = 0.5; 
sensor.piercing = False
sensor.max_range = 1000.0
sensor.angles = np.array([0])
drone.mount_sensor(sensor)

simulated_sensor = deepcopy(sensor)
simulated_sensor.piercing = True

simulated_solids = [s.clone() for s in solids]
simulation = RayCastingMap(simulated_solids)

#create world scene
visApp = VisApp()
[visApp.add_solid(s,"world") for s in world.solids]
visApp.redraw("world")
visApp.show_axes(True,"world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")

drone.sensor.bias = 0.0
drone.sensor.std = 0.000001
simulated_sensor.std = 1.0

N = 1000
history_pz_ij = np.zeros((N,3))
bias = np.linspace(-1, 6, N)
beliefs = [0.1, 1.0, 0.1]
world.update_solids_beliefs(beliefs)
for i, b in enumerate(bias):
    drone.sensor.bias = b
    z, z_ids, z_p = drone.scan(world, project_scan = True)
    simulated_z, simulated_z_ids = simulated_sensor.sense(drone.pose, simulation, 10, noisy = False)

    # pz_ij_new, temp2 = filters.new_forward_ray(z[0], simulated_z[0], simulated_z_ids[0], \
    #                     beliefs, simulated_sensor.std, simulated_sensor.max_range)
    # print(f"pz_ij_new:\n {pz_ij_new}")

    pz_ij_newnew = filters.new_new_forward_ray(z[0], simulated_z[0], simulated_z_ids[0], \
                        beliefs, simulated_sensor.std, simulated_sensor.max_range)

    history_pz_ij[i] = pz_ij_newnew
    print(f"pz_ij_newnew:\n {pz_ij_newnew}")
    # print(f"npz_ij:\n {temp1/temp2}")
    # print(f" beliefs:\n {beliefs}")

def plot_solid_on_xz(ax, solid : IfcSolid):
    v = np.asarray(solid.geometry.vertices)[:,[0,2]]
    f = np.asarray(solid.geometry.triangles)

    triangles = v[f]
    for tri in triangles:
        if tri.size == np.unique(tri, axis = 0).size: #NO IDEA WHY THIS WORKS. BUT FINE
            ax.add_patch(Polygon(tri, closed = True, 
                    color = solid.material.base_color[:3],
                    alpha = solid.material.base_color[3],
                    edgecolor = None))

fig = plt.figure()
ax = fig.add_subplot(111)
for s in world.solids:
    plot_solid_on_xz(ax, s)
xhit = np.min(np.asarray(world.solids[0].geometry.vertices)[:,0])
g_pz_1, = ax.plot(bias + xhit, history_pz_ij[:,0])
g_pz_2, = ax.plot(bias + xhit, history_pz_ij[:,1])
g_pz_3, = ax.plot(bias + xhit, history_pz_ij[:,2])
fig.legend([g_pz_1, g_pz_2, g_pz_3], ['pz_1', 'pz_2', 'pz_3'])
ax.grid(True)
plt.show()
