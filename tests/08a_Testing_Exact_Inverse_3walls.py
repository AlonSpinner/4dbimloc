import numpy as np
from numpy.matlib import repmat
from bim4loc.binaries.paths import IFC_THREE_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import IfcSolid, ifc_converter
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

np.set_printoptions(precision=3)

solids = ifc_converter(IFC_PATH)
world = RayCastingMap(solids)

drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar(angles_u = np.array([0]), angles_v = np.array([0])); sensor.std = 0.1; 
sensor.piercing = False
sensor.max_range = 1000.0
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
simulated_sensor.std = 0.5
simulated_sensor.max_range = 20.0

N = 1000
history_pz_ij = np.zeros((N,4))
bias = np.linspace(-3, 6, N)
beliefs = [0.01, 0.01, 0.01]
world.update_solids_beliefs(beliefs)
visApp.redraw("world")
for i, b in enumerate(bias):
    drone.sensor.bias = b
    z, z_ids, _, z_p = drone.scan(world, project_scan = True)
    simulated_z, simulated_z_ids, _, _, _ = simulated_sensor.sense(drone.pose, simulation, 10, noisy = False)

    pj_z_i, pz = inverse_lidar_model(z[0], simulated_z[0], simulated_z_ids[0], \
                        beliefs, simulated_sensor.std, simulated_sensor.max_range)
    history_pz_ij[i] = np.hstack((pj_z_i, pz))
    
    print(f"pz_ij:\n {history_pz_ij[:3]}")

def plot_solid_on_xz(ax, solid : IfcSolid, color):
    v = np.asarray(solid.geometry.vertices)[:,[0,2]]
    f = np.asarray(solid.geometry.triangles)

    triangles = v[f]
    for tri in triangles:
        if tri.size == np.unique(tri, axis = 0).size: #NO IDEA WHY THIS WORKS. BUT FINE
            ax.add_patch(Polygon(tri, closed = True, 
                    color = color,
                    alpha = solid.material.base_color[3],
                    edgecolor = None))

    ax.text(np.mean(v[:,0]), np.mean(v[:,1]), f" belief = {solid.material.base_color[3]}", 
                        fontsize = 10,
                        horizontalalignment='center',
                        verticalalignment='center')

fig = plt.figure()
ax = fig.add_subplot(111)
xhit = np.min(np.asarray(world.solids[0].geometry.vertices)[:,0])
g_pz_1, = ax.plot(bias + xhit, history_pz_ij[:,0], color = 'blue'); 
plot_solid_on_xz(ax, world.solids[0], color = 'blue')
g_pz_2, = ax.plot(bias + xhit, history_pz_ij[:,1], color = 'red'); 
plot_solid_on_xz(ax, world.solids[1], color = 'red')
g_pz_3, = ax.plot(bias + xhit, history_pz_ij[:,2], color = 'green'); 
plot_solid_on_xz(ax, world.solids[2], color = 'green')
g_pz, = ax.plot(bias + xhit, history_pz_ij[:,3], color = 'black')
fig.legend([g_pz_1, g_pz_2, g_pz_3, g_pz], ['pm1|z', 'pm2|z', 'pm3|z', 'g_pz'])
ax.grid(True)
plt.show()
