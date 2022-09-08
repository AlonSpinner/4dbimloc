import numpy as np
from bim4loc.geometry.poses import Pose2z
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import PcdSolid, ifc_converter
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors import Lidar
import bim4loc.existance_mapping.filters as filters
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

drone = Drone(pose = Pose2z(3,3,0, 1.5))
sensor = Lidar(angles_u = np.linspace(-np.pi/2, +np.pi/2, 36),
                 angles_v = np.linspace(-np.pi/30, +np.pi/30, 3)); 
sensor.std = 0.1
sensor.piercing = False
sensor.max_range = 100.0
drone.mount_sensor(sensor)


_, _, extent = world.bounds()
Nparticles = 100
inital_poses = []
for i in range(Nparticles):
    inital_poses.append(
        Pose2z(np.random.uniform(drone.pose.y + extent[0]/10, drone.pose.y - extent[0]/10),
                np.random.uniform(drone.pose.y + extent[1]/10, drone.pose.y - extent[1]/10),
                np.random.uniform(-np.pi, +np.pi),
                0)
    )

#create world scene
visApp = VisApp()
[visApp.add_solid(s,"world") for s in world.solids]
visApp.redraw("world")
visApp.show_axes(True,"world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")
world_scan = PcdSolid("world_scan")
visApp.add_solid(world_scan, "world")