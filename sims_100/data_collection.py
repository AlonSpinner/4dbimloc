import numpy as np
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ScanSolid, TrailSolid, ArrowSolid, \
                         update_existence_dependence_from_yaml, \
                         remove_constructed_solids_that_cant_exist
from bim4loc.geometry.raycaster import NO_HIT
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.sensors.sensors import Lidar
from bim4loc.geometry.pose2z import compose_s
from bim4loc.utils.load_yaml import load_parameters, get_actions, get_U_COV
import time
import pickle
import os
from importlib import import_module
import bim4loc.binaries.paths as ifc_paths

def create_data(seed_number, out_folder,  vis_on = False):
    np.random.seed(seed_number)

    #BUILD WORLD
    yaml_file = os.path.join(out_folder, "parameters.yaml")
    parameters_dict = load_parameters(yaml_file)
    ifc_file_path = getattr(import_module(ifc_paths.__name__),parameters_dict['IFC_PATH'])
    solids = ifc_converter(ifc_file_path)
    update_existence_dependence_from_yaml(solids, parameters_dict['existence_dependence'])
    current_time =  parameters_dict['current_time']#[s]
    constructed_solids = []
    for i, s in enumerate(solids):
        s.set_random_completion_time()
        if s.completion_time < current_time:
                constructed_solids.append(s.clone())
    constructed_solids = remove_constructed_solids_that_cant_exist(constructed_solids)

    initial_beliefs = np.zeros(len(solids))
    for i, s in enumerate(solids):
        s_simulation_belief = s.schedule.cdf(current_time)
        s.set_existance_belief_and_shader(s_simulation_belief)
        initial_beliefs[i] = s_simulation_belief
    simulation = RayCastingMap(solids)

    world = RayCastingMap(constructed_solids)

    #INITALIZE DRONE AND SENSOR
    drone = Drone(pose = np.array([3.0, 3.0, 2.0, 0.0]))
    sensor = Lidar(angles_u = eval(parameters_dict['sensor_u']),
                   angles_v = eval(parameters_dict['sensor_v'])); 
    sensor.std = parameters_dict['sensor_std']; 
    sensor.piercing = False; 
    sensor.max_range = parameters_dict['sensor_max_range']
    U_COV = get_U_COV(parameters_dict)
    drone.mount_sensor(sensor)

    #BUILDING ACTION SET
    actions = get_actions(parameters_dict)

    #DRAW
    if vis_on:
        visApp = VisApp()
        [visApp.add_solid(s,"world") for s in world.solids]# if s.ifc_type != 'IfcElectricDistributionBoard']
        visApp.redraw("world")
        visApp.setup_default_camera("world")
        visApp.add_solid(drone.solid, "world")
        vis_scan = ScanSolid("scan")
        visApp.add_solid(vis_scan, "world")
        trail_ground_truth = TrailSolid("trail_ground_truth", drone.pose[:3].reshape(1,3))
        visApp.add_solid(trail_ground_truth, "world")

        visApp.add_scene("initial_condition","world")
        [visApp.add_solid(s,"initial_condition") for s in simulation.solids]
        visApp.redraw("initial_condition")
        visApp.setup_default_camera("initial_condition")
        dead_reck_vis_arrow = ArrowSolid("dead_reck_arrow", 1.0, drone.pose)
        visApp.add_solid(dead_reck_vis_arrow, "initial_condition")
        dead_reck_vis_trail_est = TrailSolid("trail_est", drone.pose[:3].reshape(1,3))
        visApp.add_solid(dead_reck_vis_trail_est, "initial_condition")

    #measurements
    measurements = {'U' : [], 'Z' : [], 'Z_perfect' : [], 'dead_reck' : [drone.pose]}

    #ground truth
    gt_traj = [drone.pose]

    electric_boxes_names = [s.name for s in simulation.solids if s.ifc_type == 'IfcElectricDistributionBoard']
    electric_boxes_seen_counter = {name:0 for name in electric_boxes_names}
    world_solid_names = [s.name for s in world.solids]
    
    
    #LOOP
    time.sleep(2)
    for t, u in enumerate(actions):

        #move drone
        drone.move(u)
        
        #produce measurement
        z, z_ids, _, z_p = drone.scan(world, project_scan = True, n_hits = 10, noisy = True)
        z_perfect, _, _, _ = drone.scan(world, project_scan = True, n_hits = 10, noisy = False)
        for id in z_ids:
            if id != NO_HIT and world_solid_names[id] in electric_boxes_names:
                electric_boxes_seen_counter[world_solid_names[id]] += 1

        u_noisy = compose_s(u,np.random.multivariate_normal(np.zeros(4), U_COV))

        #update measurements
        gt_traj.append(drone.pose)
        dead_reck_prev = measurements['dead_reck'][-1]
        measurements['dead_reck'].append(compose_s(dead_reck_prev,u_noisy))
        measurements['U'].append(u_noisy)
        measurements['Z'].append(z)
        measurements['Z_perfect'].append(z_perfect)
        
        #updating drawings
        if vis_on:
            vis_scan.update(drone.pose[:3], z_p.T)
            visApp.update_solid(vis_scan)
            visApp.update_solid(drone.solid)
            trail_ground_truth.update(drone.pose[:3].reshape(1,-1))
            visApp.update_solid(trail_ground_truth, "world")
            visApp.redraw_all_scenes()

            dead_reck_vis_arrow.update_geometry(measurements['dead_reck'][-1] - np.array([0,0,0.3,0.0])) #wierd offset required
            dead_reck_vis_trail_est.update(measurements['dead_reck'][-1][:3].reshape(1,-1))
            visApp.update_solid(dead_reck_vis_arrow, "initial_condition")
            visApp.update_solid(dead_reck_vis_trail_est, "initial_condition")

        # time.sleep(0.1)
    measurements['dead_reck'] = np.array(measurements['dead_reck'])
    measurements['electric_boxes_seen_counter'] = electric_boxes_seen_counter

    data = {}
    data['current_time'] = current_time
    data['IFC_PATH'] = ifc_file_path
    data['sensor'] = sensor
    data['measurements'] = measurements
    data['ground_truth'] = {'constructed_solids_names': [s.name for s in constructed_solids],
                            'trajectory': np.array(gt_traj)}
    data['U_COV'] = U_COV

    file = os.path.join(out_folder, "data", f"data_{seed_number}.p")
    pickle.dump(data, open(file, "wb"))
    print('pickle dumped')