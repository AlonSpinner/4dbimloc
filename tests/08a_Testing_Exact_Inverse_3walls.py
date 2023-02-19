import numpy as np
from bim4loc.binaries.paths import IFC_THREE_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import IfcSolid, ifc_converter
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors.sensors import Lidar
from bim4loc.sensors.models import inverse_lidar_model as inverse_lidar_model
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

np.set_printoptions(precision=3)

solids = ifc_converter(IFC_PATH)
world = RayCastingMap(solids)

drone = Drone(pose = np.array([0.0, 3.0, 0.5, 0.0]))
sensor = Lidar(angles_u = np.array([0]), angles_v = np.array([0])); sensor.std = 0.05; 
sensor.piercing = False
sensor.max_range = 10.0
sensor.p0 = 0.4
drone.mount_sensor(sensor)

simulated_sensor = deepcopy(sensor)
simulated_sensor.piercing = True
simulated_sensor.max_range_cutoff = False

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
drone.sensor.std = 0.00000000001
simulated_sensor.std = 0.2

N = 1000
history_pz_ij = np.zeros((N,4))

min_x = np.min([min(s.get_vertices()[:,0]) for s in world.solids])
max_x = np.max([max(s.get_vertices()[:,0]) for s in world.solids])

beliefs = np.array([1.0, 0.5, 0.5])
world.update_solids_beliefs(beliefs)
visApp.redraw("world")
# w_z_array = np.hstack((np.linspace(0,sensor.max_range,int(N/2)),np.linspace(sensor.max_range, 10.0, int(N/2))))
w_z_array = np.linspace(0,sensor.max_range, N)
simulated_z, simulated_z_ids, _, _, _ = simulated_sensor.sense(drone.pose, simulation, 10, noisy = False)
for i, w_z in enumerate(w_z_array):
    pj_z_i, pz = inverse_lidar_model(w_z, simulated_z[0], simulated_z_ids[0], \
                        beliefs, 
                        simulated_sensor.std, simulated_sensor.max_range, simulated_sensor.p0)
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
                    alpha = 0.3,
                    edgecolor = None,
                    linewidth = 0.0))

    # ax.text(np.mean(v[:,0]), np.mean(v[:,1]), f" belief = {solid.material.base_color[3]}", 
    #                     fontsize = 10,
    #                     horizontalalignment='center',
    #                     verticalalignment='center')
plt.rcParams['font.size'] = '24'
fig = plt.figure(figsize = (16,8))
ax = fig.add_subplot(111)
ax2 = ax.twinx()
xhit = np.min(np.asarray(world.solids[0].geometry.vertices)[:,0])
g_pz_1, = ax2.plot(w_z_array, history_pz_ij[:,0], color = 'blue', linewidth = 3); 
plot_solid_on_xz(ax, world.solids[0], color = 'blue')
g_pz_2, = ax2.plot(w_z_array, history_pz_ij[:,1], color = 'red', linewidth = 3); 
plot_solid_on_xz(ax, world.solids[1], color = 'red')
g_pz_3, = ax2.plot(w_z_array, history_pz_ij[:,2], color = 'green', linewidth = 3); 
plot_solid_on_xz(ax, world.solids[2], color = 'green')
db = w_z_array[1] - w_z_array[0]
normalizer = np.sum(history_pz_ij[:,3])*db
print(normalizer)
g_pz, = ax.plot(w_z_array, history_pz_ij[:,3], color = 'black', linewidth = 3)
ax.set_xlabel('range, m', fontsize = 28)
ax.set_ylabel('probability density', fontsize = 28)
ax.set_ybound(0,1.1)
ax.set_xlim(-0.5,10.5)
ax2.set_ybound(0,1.1)
ax2.set_ylabel('probability', fontsize = 28)
ax2.set_xlim(-0.5,10.5)

# fig.legend([g_pz_1, g_pz_2, g_pz_3, g_pz], ['pm1|z', 'pm2|z', 'pm3|z', 'g_pz'])
ax.grid(True)
plt.show()
