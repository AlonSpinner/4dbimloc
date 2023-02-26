import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter
from bim4loc.rbpf.tracking.bimloc_robust import RBPF as robust
from bim4loc.rbpf.tracking.bimloc_semi_robust import RBPF as semi_robust
from bim4loc.rbpf.tracking.bimloc_simple import RBPF as simple
# from bim4loc.rbpf.tracking.bimloc_logodds_semi_robust import RBPF as logodds_semi_robust
from bim4loc.rbpf.tracking.bimloc_logodds import RBPF as logodds
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.geometry import pose2z
from bim4loc.solids import ifc_converter, ParticlesSolid, TrailSolid, ScanSolid, \
                            update_existence_dependence_from_yaml, add_variations_from_yaml
import time

A = 1
B = 4

#get data and results
dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path, "25a_data.p")
data = pickle.Unpickler(open(file, "rb")).load()
file = os.path.join(dir_path, "25b_results.p")
results = pickle.Unpickler(open(file, "rb")).load()
yaml_file = os.path.join(dir_path, "25_complementry_IFC_data.yaml")

#BUILD GROUND TRUTH
solids = ifc_converter(data['IFC_PATH'])
world_solids = [s.clone() for s in solids if s.name in data['ground_truth']['constructed_solids_names']]
sensor = data['sensor']
drone = Drone(pose = data['ground_truth']['trajectory'][0])
drone.mount_sensor(sensor)

#ADD TO VISAPP
#WORLD
visApp = VisApp()
[visApp.add_solid(s, "world") for s in world_solids]
visApp.redraw("world")
visApp.setup_default_camera("world")
vis_scan = ScanSolid("scan")
visApp.add_solid(drone.solid,"world")
visApp.add_solid(vis_scan, "world")
trail_ground_truth = TrailSolid("trail_ground_truth", drone.pose[:3].reshape(1,3))
visApp.add_solid(trail_ground_truth, "world")
#METHODS
def add_method2_visApp(N : int):
    simulation = [s.clone() for s in solids]
    add_variations_from_yaml(simulation, yaml_file)
    update_existence_dependence_from_yaml(simulation, yaml_file)
    simulation = RayCastingMap(simulation)

    visApp.add_scene(f"method{N}","world")
    [visApp.add_solid(s,f"method{N}") for s in simulation.solids]
    visApp.redraw(f"method{N}")
    visApp.setup_default_camera(f"method{N}")
    vis_particles = ParticlesSolid(poses = results[N]['particle_poses'][0])
    visApp.add_solid(vis_particles.lines, f"method{N}")
    visApp.add_solid(vis_particles.tails, f"method{N}")
    vis_trail_est = TrailSolid("trail_est", results[N]['pose_mu'][0][:3].reshape(1,3))
    visApp.add_solid(vis_trail_est, f"method{N}")
    return simulation, vis_particles, vis_trail_est

def update_method_drawings(N : int, simulation, vis_particles, vis_trail_est):
    simulation.update_solids_beliefs(results[N]['expected_belief_map'][t+1])
    [visApp.update_solid(s,f"method{N}") for s in simulation.solids]
    vis_particles.update(results[N]['particle_poses'][t+1], results[N]['particle_weights'][t+1])
    visApp.update_solid(vis_particles.lines, f"method{N}")
    visApp.update_solid(vis_particles.tails, f"method{N}")
    vis_trail_est.update(results[N]['pose_mu'][t+1][:3].reshape(1,3))
    visApp.update_solid(vis_trail_est, f"method{N}")
    visApp.redraw(f"method{N}")

visApp_A = add_method2_visApp(A)
visApp_B = add_method2_visApp(B)

for t, z in enumerate(data['measurements']['Z']):
    drone.update_pose(data['ground_truth']['trajectory'][t+1])
    z_p = pose2z.transform_from(drone.pose, drone.sensor.scan_to_points(z))

    #GROUND TRUTH DRAWINGS
    #updating drawings
    vis_scan.update(drone.pose[:3], z_p.T)
    visApp.update_solid(vis_scan)
    visApp.update_solid(drone.solid)
    trail_ground_truth.update(drone.pose[:3].reshape(1,-1))
    visApp.update_solid(trail_ground_truth, "world")
    visApp.redraw("world")
    
    #METHOD DRAWNGS
    update_method_drawings(A, *visApp_A)
    update_method_drawings(B, *visApp_B)

    visApp.redraw_all_scenes()
    # time.sleep(0.1)

