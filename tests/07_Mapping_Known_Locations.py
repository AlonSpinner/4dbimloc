import numpy as np
from bim4loc.geometry import Pose2z
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import PcdSolid, ifc_converter, PcdSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayTracingMap
import time
import logging
from bim4loc.matcher import match

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

current_time = 5.0 #[s]
solids = ifc_converter(IFC_ONLY_WALLS_PATH)

constructed_solids = []
for s in solids:
    s.set_random_completion_time()
    if s.completion_time < current_time:
        constructed_solids.append(s.clone())
world = RayTracingMap(constructed_solids)

belief_solids = [s.clone() for s in solids]
for s in belief_solids:
    s.set_shader_and_existance_belief(0.5)
belief = RayTracingMap(belief_solids)

drone = Drone(pose = Pose2z(3,3,0, 1.5));  drone.sensor.std = 0.05

straight = Pose2z(0.5,0,0,0)
turn_left = Pose2z(0,0,np.pi/8,0)
turn_right = Pose2z(0,0,-np.pi/8,0)
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

Z_STD = 0.05

#create main window
visApp = VisApp()
[visApp.add_solid(s) for s in world.solids]
visApp.show_axes(True)
visApp.reset_camera_to_default()
visApp.add_solid(drone.solid)
pcd_scan = PcdSolid()
visApp.add_solid(pcd_scan)

#create belief window
visApp.add_window('belief - view')
[visApp.add_solid(s) for s in belief.solids]
visApp.show_axes(True)
visApp.reset_camera_to_default()

time.sleep(0.1)
for t,u in enumerate(actions):
    drone.move(u)
    
    z, solid_names, z_p = drone.scan(world)
    belief_z, belief_solid_names, _ = drone.scan(belief)
    exist_solid_names, notexist_solid_names = match(z, solid_names, belief_z, belief_solid_names)

    belief.update_belief(drone.sensor.forward_existence_model, exist_solid_names, notexist_solid_names)
    
    visApp.set_active_window(1)
    [visApp.update_solid(s) for s in belief.solids]
    time.sleep(0.2)

    visApp.set_active_window(0)
    pcd_scan.update(z_p.T)
    visApp.update_solid(drone.solid)
    visApp.update_solid(pcd_scan)

    time.sleep(0.1)

print('finished')
