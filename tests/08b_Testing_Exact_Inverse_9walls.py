import numpy as np
from numpy.matlib import repmat
from bim4loc.binaries.paths import IFC_NINE_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import PcdSolid, LinesSolid, ifc_converter
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors import Lidar
from bim4loc.geometry.raycaster import NO_HIT
import bim4loc.existance_mapping.filters as filters
from copy import deepcopy
import time
import logging
import keyboard

np.set_printoptions(precision=3)

solids = ifc_converter(IFC_PATH)
world = RayCastingMap(solids)

drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar(angles_u = np.array([0]), angles_v = np.array([0])); sensor.std = 0.5; 
sensor.piercing = False
sensor.max_range = 20.0
sensor.angles = np.array([0])
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

#create belief window
visApp.add_scene("simulation", "world")
[visApp.add_solid(s,"simulation") for s in simulation.solids]
visApp.redraw("simulation")
visApp.show_axes(True,"simulation")
visApp.setup_default_camera("simulation")
visApp.redraw("simulation")

bullet = PcdSolid()
bullet.name = "bullet"
bullet.material.point_size = 15.0
bullet.material.base_color = np.array([0, 1, 1, 1])
visApp.add_solid(bullet, "world")

shot_counter = 0
move_1unit_left = np.array([0.0, 1.0, 0.0, 0.0])
drone.sensor.bias = 0.0
drone.sensor.std = 0.001
simulated_sensor.std = 0.5
while True:
    # keyboard.wait('space')
    z, z_ids, _, z_p = drone.scan(world, project_scan = True)
    simulated_z, simulated_z_ids, _, _ = simulated_sensor.sense(drone.pose, simulation, 10, noisy = False)

    for t in np.linspace(0,1,10):
        p_bullet = (1 - t) * drone.pose[:3].reshape(3,1) + t * z_p
        bullet.update(p_bullet.T)
        visApp.update_solid(bullet,"world")
        time.sleep(0.03)

    filters.exact(beliefs, z, simulated_z, simulated_z_ids, 
                    simulated_sensor.std , sensor.max_range)
    simulation.update_solids_beliefs(beliefs)

    [visApp.update_solid(s,"simulation") for s in simulation.solids]
    visApp.update_solid(drone.solid,"world")

    shot_counter += 1
    if shot_counter % 3 ==0:
        for t in range(0, 5):
            drone.move(move_1unit_left)
            visApp.update_solid(drone.solid, "world")
            time.sleep(0.1)

    visApp.redraw_all_scenes()
