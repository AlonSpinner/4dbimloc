import numpy as np
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import PcdSolid, ifc_converter, ScanSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors.sensors import Lidar
from bim4loc.geometry.pose2z import compose_s, transform_from
import bim4loc.existance_mapping.filters as filters
from copy import deepcopy
import time
import logging
import keyboard

np.random.seed(24)

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

current_time = 5.0 #[s]
solids = ifc_converter(IFC_PATH)

constructed_solids = []
for s in solids:
    s.set_random_completion_time()
    if s.completion_time < current_time:
        constructed_solids.append(s.clone())
world = RayCastingMap(constructed_solids)

drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar(angles_u = np.linspace(-np.pi, np.pi, 100), angles_v = np.array([0])); 
sensor.std = 0.1; 
sensor.piercing = False
sensor.max_range = 100.0
drone.mount_sensor(sensor)

simulated_sensor = deepcopy(sensor)
simulated_sensor.piercing = True
simulated_sensor.std = 1 * sensor.std

simulation_solids = [s.clone() for s in solids]
simulation = RayCastingMap(simulation_solids)
beliefs = np.full(len(simulation.solids), 0.5)
simulation.update_solids_beliefs(beliefs)

straight = np.array([0.5,0.0 ,0.0 ,0.0])
turn_left = np.array([0.0 ,0.0 ,0.0, np.pi/8])
turn_right = np.array([0.0, 0.0, 0.0, -np.pi/8])
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

#create world scene
visApp = VisApp()
[visApp.add_solid(s,"world") for s in world.solids]
visApp.redraw("world")
visApp.show_axes(True,"world")
visApp.setup_default_camera("world")
# visApp.add_solid(drone.solid, "world")
vis_scan = ScanSolid("scan")
visApp.add_solid(vis_scan, "world")

simulated_drone = Drone(drone.pose)
simulated_drone.solid.name = "simulated_drone"
simulated_drone.solid.material.base_color = np.array([0, 1, 0, 1])
# visApp.add_solid(simulated_drone.solid, "world")
simulated_vis_scan = ScanSolid("simulated_scan", color = np.array([1.0, 0.0, 0.8]))
visApp.add_solid(simulated_vis_scan, "world")

#create simulation window
visApp.add_scene("simulation", "world")
[visApp.add_solid(s,"simulation") for s in simulation.solids]
visApp.redraw("simulation")
visApp.show_axes(True,"simulation")
visApp.setup_default_camera("simulation")

time.sleep(1)
dt = 0.0
start_time = time.time()
for t,u in enumerate(actions):
    keyboard.wait('space')
    step_start = time.time()
    
    drone.move(u)
    test_pose = compose_s(drone.pose, np.array([0.1,0.0,0.0,0.0]))
    simulated_drone.solid.update_geometry(test_pose)

    z, z_ids, _, z_p  = drone.scan(world, project_scan = True, noisy = False)
    
    simulated_z, simulated_z_ids, _, _, _ = simulated_sensor.sense(test_pose, simulation, 10, noisy = False)
    #scan match?
    fixed_pose = scan_match(z, simulated_z)
    simulated_z, simulated_z_ids, _, _, _ = simulated_sensor.sense(test_pose, simulation, 10, noisy = False)


    filters.exact2(beliefs, z, simulated_z, simulated_z_ids, 
                    sensor.std, sensor.max_range)
    simulation.update_solids_beliefs(beliefs)
    
    vis_scan.update(drone.pose[:3], z_p.T)
    
    simulated_first_hits = simulated_z[:,0]
    simulated_drone_p = simulated_sensor.scan_to_points(simulated_first_hits)
    simulated_world_p = transform_from(test_pose,simulated_drone_p)
    simulated_vis_scan.update(test_pose[:3], simulated_world_p.T)

    [visApp.update_solid(s,"simulation") for s in simulation.solids]
    # visApp.update_solid(drone.solid,"world")
    # visApp.update_solid(simulated_drone.solid,"world")
    visApp.update_solid(vis_scan, "world")
    visApp.update_solid(simulated_vis_scan, "world")
    
    visApp.redraw_all_scenes()
    
    # step_end = time.time()
    # time.sleep(max(dt - (step_end - step_start),0))

end_time = time.time()
print(f'finished in {end_time - start_time}')
visApp.redraw_all_scenes()