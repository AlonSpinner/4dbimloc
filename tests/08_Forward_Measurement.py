import numpy as np
import numpy.matlib
from bim4loc.geometry.poses import Pose2z
from bim4loc.binaries.paths import IFC_THREE_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import PcdSolid, LinesSolid, ifc_converter
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors import Lidar1D
from bim4loc.random.utils import p2logodds
from bim4loc.geometry.raycaster import NO_HIT
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
sensor.max_range = 20.0
sensor.angles = np.array([0, 0.1])
drone.mount_sensor(sensor)

simulated_sensor = deepcopy(sensor)
simulated_sensor.piercing = True

simulated_solids = [s.clone() for s in solids]
simulation = RayCastingMap(simulated_solids)
beliefs = np.full(len(simulation.solids), 0.5)
simulation.update_solids_beliefs(beliefs)

#create world scene
visApp = VisApp()
[visApp.add_solid(s,"world") for s in world.solids]
visApp.redraw("world")
visApp.show_axes(True,"world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")
pcd_scan_world = PcdSolid()
visApp.add_solid(pcd_scan_world, "world")

#create belief window
visApp.add_scene("simulation", "world")
[visApp.add_solid(s,"simulation") for s in simulation.solids]
visApp.redraw("simulation")
visApp.show_axes(True,"simulation")
visApp.setup_default_camera("simulation")
visApp.redraw("simulation")
line_scan = LinesSolid()
visApp.add_solid(line_scan, "simulation")
pcd_scan_simulation = PcdSolid()
visApp.add_solid(pcd_scan_world, "simulation")

# keyboard.wait('space')

def calcualte_lines(simulated_z, angles, drone_pose):
    z_flat = simulated_z.flatten()
    angles_flat = np.matlib.repmat(angles, 1 , simulated_z.shape[1]).flatten()

    drone_p = np.vstack((z_flat * np.cos(angles_flat), 
                    z_flat * np.sin(angles_flat),
                    np.zeros_like(z_flat)))
    world_p = drone_pose.transform_from(drone_p)

    world_p = np.hstack((drone_pose.t, world_p))
    line_ids = np.zeros((world_p.shape[1],2), dtype = int)
    line_ids[:,1] = np.arange(world_p.shape[1])
    return world_p, line_ids

while True:
    z, z_ids, z_p = drone.scan(world, project_scan = True)
    simulated_z, simulated_z_ids = simulated_sensor.sense(drone.pose, simulation, 10)

    print(filters.pz(z[0], simulated_z[0], simulated_z_ids[0], beliefs, sensor.std))
    simulation.update_solids_beliefs(beliefs)
    
    pcd_scan_world.update(z_p.T)
    line_p, line_ids = calcualte_lines(simulated_z, simulated_sensor.angles, drone.pose)
    line_scan.update(line_p.T, line_ids)
    pcd_scan_simulation.update(line_p.T)

    [visApp.update_solid(s,"simulation") for s in simulation.solids]
    visApp.update_solid(drone.solid,"world")
    visApp.update_solid(pcd_scan_world,"world")
    visApp.update_solid(line_scan, "simulation")
    visApp.update_solid(pcd_scan_simulation,"simulation")
    
    visApp.redraw_all_scenes()

    keyboard.wait('space')

print('finished')
visApp.redraw_all_scenes()