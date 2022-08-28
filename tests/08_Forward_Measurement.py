import numpy as np
from bim4loc.geometry.poses import Pose2z
from bim4loc.binaries.paths import IFC_THREE_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import PcdSolid, ifc_converter
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors import Lidar1D
from bim4loc.random.utils import p2logodds
import bim4loc.existance_mapping.filters as filters
from copy import deepcopy
import time
import logging
import keyboard

solids = ifc_converter(IFC_PATH)
world = RayCastingMap(solids)

drone = Drone(pose = Pose2z(3.0,2.5, 0, 1.5))
sensor = Lidar1D(); sensor.std = 0.05; 
sensor.piercing = False
sensor.max_range = 100.0
sensor.angles = np.array([0])
drone.mount_sensor(sensor)

simulated_sensor = deepcopy(sensor)
simulated_sensor.piercing = True

belief_solids = [s.clone() for s in solids]
belief = RayCastingMap(belief_solids)
logodds_beliefs = np.full(len(belief.solids), p2logodds(0.5))
belief.update_solids_beliefs(logodds_beliefs)

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
visApp.redraw("belief")

keyboard.wait('space')
while True:
    z, z_ids, z_p = drone.scan(world, project_scan = True)
    simulated_z, simulated_z_ids = simulated_sensor.sense(drone.pose, belief, 10)

    filters.vanila_inverse(logodds_beliefs, z, simulated_z, simulated_z_ids, sensor.std, sensor.max_range)
    belief.update_solids_beliefs(logodds_beliefs)
    
    pcd_scan.update(z_p.T)

    [visApp.update_solid(s,"belief") for s in belief.solids]
    visApp.update_solid(drone.solid,"world")
    visApp.update_solid(pcd_scan,"world")
    
    visApp.redraw_all_scenes()

    keyboard.wait('space')

print('finished')
visApp.redraw_all_scenes()