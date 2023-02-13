import numpy as np
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import PcdSolid, ifc_converter
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors.sensors import Lidar
from bim4loc.geometry.pose2z import compose_s, s_from_Rt
from copy import deepcopy, copy
import time
import logging
import keyboard
import bim4loc.geometry.scan_matcher.scan_matcher as scan_matcher

np.random.seed(25)

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

current_time = 5.0 #[s]
solids = ifc_converter(IFC_PATH)

constructed_solids = []
for s in solids:
    s.set_random_completion_time()
    # if s.completion_time < current_time:
    constructed_solids.append(s.clone())
world = RayCastingMap(constructed_solids)

drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar(angles_u = np.linspace(-np.pi/2, +np.pi/2, 100),
                 angles_v = np.array([0])); 
sensor.std = 0.1
sensor.piercing = False
sensor.max_range = 100.0
drone.mount_sensor(sensor)

simulated_drone = Drone(copy(drone.pose))
simulated_sensor = Lidar(angles_u = np.linspace(-np.pi/2, +np.pi/2, 100),
                 angles_v = np.array([0])); 
simulated_sensor.max_range = 100.0
# simulated_sensor = deepcopy(sensor)
simulated_sensor.piercing = True
simulated_sensor.max_range_cutoff = False
simulated_drone.mount_sensor(simulated_sensor)
simulation_solids = [s.clone() for s in solids]
simulation = RayCastingMap(simulation_solids)
beliefs = np.full(len(simulation.solids), 0.5)
simulation.update_solids_beliefs(beliefs)

straight = np.array([0.5,0.0 ,0.0 ,0.0])
turn_left = np.array([0.0 ,0.0 ,0.0, np.pi/8])
turn_right = np.array([0.0, 0.0, 0.0, -np.pi/8])
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20 + 4 * [turn_right]

#create world scene
visApp = VisApp()
[visApp.add_solid(s,"world") for s in world.solids]
visApp.redraw("world")
visApp.show_axes(True,"world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")
world_scan = PcdSolid("world_scan")
visApp.add_solid(world_scan, "world")

#create simulation window
visApp.add_scene("simulation", "world")
[visApp.add_solid(s,"simulation") for s in simulation.solids]
visApp.redraw("simulation")
visApp.show_axes(True,"simulation")
visApp.setup_default_camera("simulation")
visApp.add_solid(simulated_drone.solid, "simulation")
simulation_scan = PcdSolid("simulation_scan")
visApp.add_solid(simulation_scan, "simulation")
visApp.redraw("simulation")

time.sleep(1)
dt = 0.0
for t,u in enumerate(actions):
    # keyboard.wait('space')
    step_start = time.time()
    
    drone.move(u)
    
    z, z_ids, _, z_p = drone.scan(world, project_scan = True, noisy = False)

    errT = np.array([0.5, 0.0, 0.0, 5 *np.pi/180])
    simulated_drone.pose = compose_s(drone.pose,errT)
    simulated_drone.solid.update_geometry(simulated_drone.pose)

    simulated_z, simulated_z_ids, simulated_z_normals, simulated_z_p = simulated_drone.scan(simulation, project_scan = True, noisy = False)

    world_scan.update(z_p.T)
    simulation_scan.update(simulated_z_p.T)
    visApp.update_solid(simulated_drone.solid,"simulation")
    visApp.update_solid(drone.solid,"world")
    visApp.update_solid(world_scan,"world")
    visApp.update_solid(simulation_scan,"simulation")

    visApp.redraw_all_scenes()
    R, t, rmse = scan_matcher.scan_match(z, simulated_z, simulated_z_ids,
                beliefs, 
                sensor.std, sensor.max_range,
                sensor.get_scan_to_points(),
                simulated_sensor.get_scan_to_points(),
                downsample_voxelsize = 0.5,
                icp_distance_threshold = 3.0,
                probability_filter_threshold = 0.3
                )

    fixed_pose = compose_s(simulated_drone.pose, s_from_Rt(R,t))
    print(f"chordal fixed pose - real pose = {fixed_pose - drone.pose}")
    
    simulation.update_solids_beliefs(beliefs)
    [visApp.update_solid(s,"simulation") for s in simulation.solids]
    visApp.redraw_all_scenes()
    
    step_end = time.time()
    time.sleep(max(dt - (step_end - step_start),0))

print('finished')
visApp.redraw_all_scenes()