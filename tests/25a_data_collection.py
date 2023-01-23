import numpy as np
from bim4loc.binaries.paths import IFC_ARENA_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ScanSolid, TrailSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors.sensors import Lidar
from bim4loc.geometry.pose2z import compose_s
import time
import logging
import pickle
import os

np.random.seed(25)
logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

#BUILD WORLD
current_time = 5.0 #[s]
solids = ifc_converter(IFC_PATH)

constructed_solids = []
for s in solids:
    s.set_random_completion_time()
    if s.completion_time < current_time:
        constructed_solids.append(s.clone())
world = RayCastingMap(constructed_solids)
solids_completion_times = np.array([s.completion_time for s in solids])

#INITALIZE DRONE AND SENSOR
drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar(angles_u = np.linspace(-np.pi,np.pi, int(300)), angles_v = np.array([0.0])); 
sensor.std = 0.1; sensor.piercing = False; sensor.max_range = 10.0
drone.mount_sensor(sensor)

#BUILDING ACTION SET
DT = 1.0
straight = np.array([0.5,0.0 ,0.0 ,0.0]) * DT
turn_left = np.array([0.0 ,0.0 ,0.0, np.pi/8]) * DT
turn_right = np.array([0.0, 0.0, 0.0, -np.pi/8]) * DT
stay = np.zeros(4) * DT
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20 + [turn_right] * 4 + [straight] * 4

#DRAW
visApp = VisApp()
[visApp.add_solid(s,"world") for s in world.solids]
visApp.redraw("world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")
vis_scan = ScanSolid("scan")
visApp.add_solid(vis_scan, "world")
trail_ground_truth = TrailSolid("trail_ground_truth", drone.pose[:3].reshape(1,3))
visApp.add_solid(trail_ground_truth, "world")

U_COV = np.diag([0.1, 0.05, 1e-25, np.radians(1.0)])/10

#measurements
measurements = {'U' : [], 'Z' : []}

#ground truth
gt_traj = []

#LOOP
time.sleep(2)
for t, u in enumerate(actions):

    #move drone
    drone.move(u)
    
    #produce measurement
    z, _, _, z_p = drone.scan(world, project_scan = True, n_hits = 5, noisy = True)

    u_noisy = compose_s(u,np.random.multivariate_normal(np.zeros(4), U_COV))

    #update measurements
    gt_traj.append(drone.pose)
    measurements['U'].append(u_noisy)
    measurements['Z'].append(z)
    
    #updating drawings
    vis_scan.update(drone.pose[:3], z_p.T)
    visApp.update_solid(vis_scan)
    visApp.update_solid(drone.solid)
    trail_ground_truth.update(drone.pose[:3].reshape(1,-1))
    visApp.update_solid(trail_ground_truth, "world")
    visApp.redraw_all_scenes()

data = {}
data['current_time'] = current_time
data['IFC_PATH'] = IFC_PATH
data['sensor'] = sensor
data['measurements'] = measurements
data['ground_truth'] = {'solids_completion_times': solids_completion_times,
                        'trajectory': np.array(gt_traj)}
data['U_COV'] = U_COV

dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path, "25a_data.p")
pickle.dump(data, open(file, "wb"))
print('pickle dumped')