import numpy as np
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid, TrailSolid, ScanSolid, \
                            update_existence_dependence_from_yaml, add_variations_from_yaml, \
                            ArrowSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.geometry import pose2z
import logging
import pickle
import os

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))
yaml_file = os.path.join(dir_path, "25_complementry_IFC_data.yaml")
data_file = os.path.join(dir_path, "25a_data.p")
data = pickle.Unpickler(open(data_file, "rb")).load()

solids = ifc_converter(data['IFC_PATH'])

constructed_solids = [s.clone() for s in solids if s.name in data['ground_truth']['constructed_solids_names']]
world = RayCastingMap(constructed_solids)

simulation_solids = solids
add_variations_from_yaml(simulation_solids, yaml_file)
update_existence_dependence_from_yaml(simulation_solids, yaml_file)    
initial_beliefs = np.zeros(len(simulation_solids))
for i, s in enumerate(simulation_solids):
    s_simulation_belief = s.schedule.cdf(data['current_time'])
    s.set_existance_belief_and_shader(s_simulation_belief)
    initial_beliefs[i] = s_simulation_belief
simulation = RayCastingMap(simulation_solids)

pose0 = data['ground_truth']['trajectory'][0]
drone = Drone(pose = pose0)
drone.mount_sensor(data["sensor"])
visApp = VisApp()
[visApp.add_solid(s,"world") for s in constructed_solids]
visApp.redraw("world")
visApp.setup_default_camera("world")
visApp.add_solid(drone.solid, "world")
vis_scan = ScanSolid("scan")
visApp.add_solid(vis_scan, "world")
z_p = pose2z.transform_from(drone.pose, drone.sensor.scan_to_points(data['measurements']['Z'][0]))
vis_scan.update(drone.pose[:3], z_p.T)
visApp.update_solid(vis_scan)
dead_reck_vis_arrow = ArrowSolid("dead_reck_arrow", 1.0, drone.pose)
visApp.add_solid(dead_reck_vis_arrow, "world")
dead_reck_vis_trail_est = TrailSolid("trail_est", 
                                        drone.pose[:3].reshape(1,3),
                                        color = [0.0,0.0,1.0])
visApp.add_solid(dead_reck_vis_trail_est, "world")
#----------------------SIMULATION-------------------------------
visApp.add_scene("simulation","world")
[visApp.add_solid(s,"simulation") for s in simulation.solids]
visApp.redraw("simulation")
visApp.setup_default_camera("simulation")
N_particles = 10
initial_particle_poses = np.random.multivariate_normal(pose0, data['U_COV'], N_particles)
sim_vis_particles = ParticlesSolid(poses = initial_particle_poses)
visApp.add_solid(sim_vis_particles.lines, "simulation")
visApp.add_solid(sim_vis_particles.tails, "simulation")

visApp.redraw_all_scenes()

def crop_image(image, crop_ratio_w, crop_ratio_h):
    h,w = image.shape[:2]
    crop_h = int(h * crop_ratio_h/2)
    crop_w = int(w * crop_ratio_w/2)
    return image[crop_h:-crop_h, crop_w:-crop_w,:]
images_output_path = os.path.join(dir_path, "25_images")

images = visApp.get_images(images_output_path,prefix = f"init_",
                transform = lambda x: crop_image(x,0.3,0.55))