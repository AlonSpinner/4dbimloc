import numpy as np
from bim4loc.binaries.paths import IFC_ARENA_PLUS_PATH as IFC_PATH
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

np.random.seed(14)
logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

#BUILD WORLD
current_time = 5.0 #[s]
solids = ifc_converter(IFC_PATH)

solids_existence_dependence = {40 : 14, 41: 14, 42: 14,
                               37: 21, 48: 21, 47: 21,
                               38: 1, 45: 1, 46: 1,
                               39: 2, 43: 2, 44: 2}
solids_varaition_dependence = np.array([[40, 41, 42], 
                                        [37, 48, 47], 
                                        [38, 45, 46], 
                                        [39, 43, 44]], dtype = int)

constructed_solids = []
duplicate_solids = solids_varaition_dependence[:, 1:].flatten()
for i, s in enumerate(solids):
    if i in duplicate_solids:
        continue
    s.set_random_completion_time()
    
    if s.completion_time < current_time: #think if solid should be constructed
        if i in solids_existence_dependence.keys():
            if solids[solids_existence_dependence[i]].completion_time < current_time: #assumes order...
                constructed_solids.append(s.clone())        
        else:
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

U_COV = np.diag([0.1, 0.05, 1e-25, np.radians(1.0)])

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
data['solids_existence_dependence'] = solids_existence_dependence
data['solids_varaition_dependence'] = solids_varaition_dependence
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