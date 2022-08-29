import numpy as np
import numpy.matlib
from bim4loc.geometry.poses import Pose2z
from bim4loc.binaries.paths import IFC_NINE_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import PcdSolid, LinesSolid, ifc_converter
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors import Lidar1D
from bim4loc.geometry.raycaster import NO_HIT
import bim4loc.existance_mapping.filters as filters
from copy import deepcopy
import time
import logging
import keyboard

solids = ifc_converter(IFC_PATH)
world = RayCastingMap(solids)

drone = Drone(pose = Pose2z(3.0,2.5, 0, 1.5))
sensor = Lidar1D(); sensor.std = 0.5; 
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

def calcualte_lines(simulated_z, angles, drone_pose):
    z_flat = simulated_z.flatten()
    angles_flat = np.matlib.repmat(angles, simulated_z.shape[1] , 1).T.flatten()

    drone_p = np.vstack((z_flat * np.cos(angles_flat), 
                    z_flat * np.sin(angles_flat),
                    np.zeros_like(z_flat)))
    world_p = drone_pose.transform_from(drone_p)

    world_p = np.hstack((drone_pose.t, world_p))
    line_ids = np.zeros((world_p.shape[1],2), dtype = int)
    line_ids[:,1] = np.arange(world_p.shape[1])
    return world_p, line_ids

bullet = PcdSolid()
bullet.name = "bullet"
bullet.material.point_size = 15.0
bullet.material.base_color = np.array([1, 0, 1, 1])
visApp.add_solid(bullet, "world")

shot_counter = 0
move_1unit_left = Pose2z(0.0, 1.0, 0.0, 0.0)
while True:
    keyboard.wait('space')
    z, z_ids, z_p = drone.scan(world, project_scan = True)
    simulated_z, simulated_z_ids = simulated_sensor.sense(drone.pose, simulation, 10, noisy = False)

    for t in np.linspace(0,1,10):
        p_bullet = (1 - t) * drone.pose.t + t * z_p
        bullet.update(p_bullet.T)
        visApp.update_solid(bullet,"world")
        time.sleep(0.1)

    filters.vanila_forward(beliefs, z, simulated_z, simulated_z_ids, sensor.std, sensor.max_range)
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

    shot_counter += 1
    if shot_counter % 3 ==0:
        for t in range(0, 5):
            drone.move(move_1unit_left)
            visApp.update_solid(drone.solid, "world")
            time.sleep(0.1)

    visApp.redraw_all_scenes()
