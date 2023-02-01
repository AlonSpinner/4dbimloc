import numpy as np
from bim4loc.binaries.paths import IFC_ARENA_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ScanSolid, TrailSolid, ArrowSolid, \
                         update_existence_dependence_from_yaml, remove_constructed_solids_that_cant_exist
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors.sensors import Lidar
from bim4loc.geometry.pose2z import compose_s
import time
import logging
import pickle
import os
dead_reck_show = True
np.random.seed(55)
#8 is simple
#5, 10, 55 are rough
#14 is good
logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

#BUILD WORLD
current_time = 5.0 #[s]
solids = ifc_converter(IFC_PATH)
dir_path = os.path.dirname(os.path.realpath(__file__))
yaml_file = os.path.join(dir_path, "25_complementry_IFC_data.yaml")
update_existence_dependence_from_yaml(solids, yaml_file)
constructed_solids = []
for i, s in enumerate(solids):
    s.set_random_completion_time()
    if s.completion_time < current_time:
            constructed_solids.append(s.clone())
constructed_solids = remove_constructed_solids_that_cant_exist(constructed_solids)
# del constructed_solids[2]

initial_beliefs = np.zeros(len(solids))
for i, s in enumerate(solids):
    s_simulation_belief = s.schedule.cdf(current_time)
    s.set_existance_belief_and_shader(s_simulation_belief)
    initial_beliefs[i] = s_simulation_belief
simulation = RayCastingMap(solids)

world = RayCastingMap(constructed_solids)
solids_completion_times = np.array([s.completion_time for s in solids])

#INITALIZE DRONE AND SENSOR
drone = Drone(pose = np.array([3.0, 3.0, 2.0, 0.0]))
sensor = Lidar(angles_u = np.linspace(-np.pi,np.pi, int(300)), angles_v = np.array([0.0])); 
sensor.std = 0.05; sensor.piercing = False; sensor.max_range = 10.0
drone.mount_sensor(sensor)

#BUILDING ACTION SET
DT = 1.0
straight = np.array([0.5,0.0 ,0.0 ,0.0]) * DT
turn_left = np.array([0.0 ,0.0 ,0.0, np.pi/8]) * DT
turn_right = np.array([0.0, 0.0, 0.0, -np.pi/8]) * DT
stay = np.zeros(4) * DT
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20 + [turn_right] * 4 + [straight] * 4

#DRAW
#0UexuGkbH1jBE7iWw9bUj6
visApp = VisApp()
[visApp.add_solid(s,"world") for s in world.solids]# if s.ifc_type != 'IfcElectricDistributionBoard']
visApp.redraw("world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")
vis_scan = ScanSolid("scan")
visApp.add_solid(vis_scan, "world")
trail_ground_truth = TrailSolid("trail_ground_truth", drone.pose[:3].reshape(1,3))
visApp.add_solid(trail_ground_truth, "world")

visApp.add_scene("initial_condition","world")
[visApp.add_solid(s,"initial_condition") for s in simulation.solids]
visApp.redraw("initial_condition")
visApp.setup_default_camera("initial_condition")

if dead_reck_show == True:
    dead_reck_vis_arrow = ArrowSolid("dead_reck_arrow", 1.0, drone.pose)
    visApp.add_solid(dead_reck_vis_arrow, "world")
    dead_reck_vis_trail_est = TrailSolid("trail_est", 
                                         drone.pose[:3].reshape(1,3),
                                         color = [0.0,0.0,1.0])
    visApp.add_solid(dead_reck_vis_trail_est, "world")

U_COV = np.diag([0.2, 0.1, 1e-25, np.radians(1)])**2

#measurements
measurements = {'U' : [], 'Z' : [], 'dead_reck' : [drone.pose]}

#ground truth
gt_traj = [drone.pose]

def crop_image(image, crop_ratio_w, crop_ratio_h):
    h,w = image.shape[:2]
    crop_h = int(h * crop_ratio_h/2)
    crop_w = int(w * crop_ratio_w/2)
    return image[crop_h:-crop_h, crop_w:-crop_w,:]
images_output_path = os.path.join(dir_path, "25_images")

#LOOP
time.sleep(2)
for t, u in enumerate(actions):

    #move drone
    drone.move(u)
    
    #produce measurement
    z, _, _, z_p = drone.scan(world, project_scan = True, n_hits = 10, noisy = True)

    u_noisy = compose_s(u,np.random.multivariate_normal(np.zeros(4), U_COV))

    #update measurements
    gt_traj.append(drone.pose)
    dead_reck_prev = measurements['dead_reck'][-1]
    measurements['dead_reck'].append(compose_s(dead_reck_prev,u_noisy))
    measurements['U'].append(u_noisy)
    measurements['Z'].append(z)
    
    #updating drawings
    vis_scan.update(drone.pose[:3], z_p.T)
    visApp.update_solid(vis_scan)
    visApp.update_solid(drone.solid)
    trail_ground_truth.update(drone.pose[:3].reshape(1,-1))
    visApp.update_solid(trail_ground_truth, "world")
    visApp.redraw_all_scenes()

    if dead_reck_show == True:
        dead_reck_vis_arrow.update_geometry(measurements['dead_reck'][-1] - np.array([0,0,0.3,0.0])) #wierd offset required
        dead_reck_vis_trail_est.update(measurements['dead_reck'][-1][:3].reshape(1,-1))
        visApp.update_solid(dead_reck_vis_arrow, "world")
        visApp.update_solid(dead_reck_vis_trail_est, "world")

    if t % 3 == 0:
        images = visApp.get_images(images_output_path,prefix = f"t{t}_",
                            transform = lambda x: crop_image(x,0.3,0.55),
                            save_scenes = "world")
    
    # time.sleep(0.1)
measurements['dead_reck'] = np.array(measurements['dead_reck'])

data = {}
data['current_time'] = current_time
data['IFC_PATH'] = IFC_PATH
data['sensor'] = sensor
data['measurements'] = measurements
data['ground_truth'] = {'constructed_solids_names': [s.name for s in constructed_solids],
                        'trajectory': np.array(gt_traj)}
data['U_COV'] = U_COV

dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path, "25a_data.p")
pickle.dump(data, open(file, "wb"))
print('pickle dumped')