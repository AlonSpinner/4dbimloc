import numpy as np
from bim4loc.geometry import Pose2z
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import PcdSolid, ifc_converter, PcdSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayTracingMap
from bim4loc.sensors import Lidar1D
from bim4loc.existance_mapping.matchers import lidar1D_matcher
from bim4loc.existance_mapping.filters import vanila_filter
import time
import logging

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
    s.set_existance_belief_and_shader(0.5)
belief = RayTracingMap(belief_solids)

drone = Drone(pose = Pose2z(3,3,0, 1.5))
sensor = Lidar1D(); sensor.std = 0.05
drone.mount_sensor(sensor)

straight = Pose2z(0.5,0,0,0)
turn_left = Pose2z(0,0,np.pi/8,0)
turn_right = Pose2z(0,0,-np.pi/8,0)
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

Z_STD = 0.05

#create world scene
visApp = VisApp()
[visApp.add_solid(s,"world") for s in world.solids]
visApp.redraw("world")
visApp.show_axes(True,"world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")
pcd_scan = PcdSolid()
visApp.add_solid(pcd_scan, "world")

#create belief window
visApp.add_scene("belief", "world")
[visApp.add_solid(s,"belief") for s in belief.solids]
visApp.redraw("belief")
visApp.show_axes(True,"belief")
visApp.setup_default_camera("belief")

time.sleep(0.1)
for t,u in enumerate(actions):
    
    drone.move(u)
    
    z, solid_names, z_p = drone.scan(world)
    belief_z, belief_solid_names, _ = drone.scan(belief)
    exist_solid_names, notexist_solid_names = lidar1D_matcher(z, belief_z, belief_solid_names, sensor.std)

    vanila_filter(belief, drone.sensor.forward_existence_model, exist_solid_names, notexist_solid_names)
    
    pcd_scan.update(z_p.T)

    [visApp.update_solid(s,"belief") for s in belief.solids]
    visApp.update_solid(drone.solid,"world")
    visApp.update_solid(pcd_scan,"world")

    time.sleep(0.1)

print('finished')
