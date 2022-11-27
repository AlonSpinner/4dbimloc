import numpy as np
from bim4loc.binaries.paths import IFC_THREE_WALLS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ScanSolid, ParticlesSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors.sensors import Lidar
from bim4loc.sensors.models import inverse_lidar_model
from bim4loc.geometry.pose2z import compose_s, transform_from
from bim4loc.geometry.raycaster import NO_HIT
import bim4loc.existance_mapping.filters as filters
from copy import deepcopy
import time
import keyboard

solids = ifc_converter(IFC_PATH)
solids = [solids[0]]
world = RayCastingMap(solids)

#INITALIZE DRONE AND SENSOR
drone = Drone(pose = np.array([10.0, 2.5, 1.5, 0.0]))
sensor = Lidar(angles_u = np.linspace(-np.pi/2,np.pi/2, 50), angles_v = np.array([0.0])); 
sensor.std = 0.1; sensor.piercing = False; sensor.max_range = 100.0
drone.mount_sensor(sensor)

simulated_sensor = deepcopy(sensor)
simulated_sensor.piercing = True
simulated_sensor.std = simulated_sensor.std * 5
simulated_solids = [s.clone() for s in solids]
[s.set_existance_belief_and_shader(0.5) for s in simulated_solids]

simulation = RayCastingMap(simulated_solids)
particle_poses = np.array([[10 +0.5, 2.5, 1.5, 0.0]])
particle_beliefs = 0.5*np.ones(len(simulation.solids))

#DRAW
visApp = VisApp()
[visApp.add_solid(s,"world") for s in simulation.solids]
visApp.redraw("world")
visApp.show_axes(True,"world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")
vis_scan = ScanSolid("scan")
visApp.add_solid(vis_scan, "world")
vis_particles = ParticlesSolid(poses = particle_poses)
visApp.add_solid(vis_particles.lines, "world")
visApp.add_solid(vis_particles.tails, "world")
particle_vis_scan = ScanSolid("simulated_scan", color = np.array([1.0, 0.0, 0.8]))
visApp.add_solid(particle_vis_scan, "world")

sense_fcn = lambda x: simulated_sensor.sense(x, simulation, n_hits = 10, noisy = False)
for time_step in range(10):
    # keyboard.wait('space')

    #produce measurement
    z, z_ids, z_normals, z_p = drone.scan(world, project_scan = True, 
                                                 noisy = True, 
                                                 n_hits = 5)

    #-----------------------FILTER------------------------------------------------
    #sense
    particle_z_values, particle_z_ids, _, _, _ = sense_fcn(particle_poses[0])

    #remap and calcualte probability of rays pz
    particle_beliefs, pz = filters.exact(particle_beliefs, 
                                    z, 
                                    particle_z_values, 
                                    particle_z_ids, 
                                    simulated_sensor.std,
                                    simulated_sensor.max_range)
  

    simulation.update_solids_beliefs(particle_beliefs) 

    #updating drawings
    vis_scan.update(drone.pose[:3], z_p.T)
    vis_particles.update(particle_poses, np.array([1.0]))
    visApp.update_solid(vis_scan)
    visApp.update_solid(drone.solid)
    visApp.update_solid(vis_particles.lines, "world")
    visApp.update_solid(vis_particles.tails, "world")
    [visApp.update_solid(s,"world") for s in simulation.solids]

    #simulated rays
    simulated_first_hits = particle_z_values[:,0]    
    simulated_drone_p = simulated_sensor.scan_to_points(simulated_first_hits)
    simulated_world_p = transform_from(particle_poses[0],simulated_drone_p)
    particle_vis_scan.update(particle_poses[0][:3], simulated_world_p.T)
    visApp.update_solid(particle_vis_scan, "world")

    print(time_step, particle_beliefs)
