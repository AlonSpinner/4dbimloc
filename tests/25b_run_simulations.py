import numpy as np
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid, TrailSolid, ScanSolid
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.geometry import pose2z
import time
import logging
import pickle
import os
from copy import deepcopy
from bim4loc.rbpf.tracking.bimloc import RBPF as RBPF_0
from bim4loc.rbpf.tracking.bimloc_partial import RBPF as RBPF_1
from bim4loc.rbpf.tracking.bimloc_logodds import RBPF as RBPF_2

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path, "25a_data.p")
data = pickle.Unpickler(open(file, "rb")).load()

results = {0: {}, 1: {}, 2: {}}
for rbpf_enum, RBPF in enumerate([RBPF_0, RBPF_1, RBPF_2]):

    #BUILD SIMULATION ENVIORMENT
    simulation_solids = ifc_converter(data['IFC_PATH'])

    #--------------------------------------create solids_varaition_dependence----------------------------
    electric_boards = [s for s in solids if s.ifc_type == 'IfcElectricDistributionBoard']
    duplicate_solids = []
    #create new solids and add the appropiate translations
    translations = [-1, 1]
    for s in electric_boards:
        for t in translations:
            s_new = s.clone()
            verts = s_new.get_vertices() + np.array([0.0,0.0,0.1]) * t
            s_new.set_vertices(verts)
            solids.append(s_new)
            duplicate_solids.append(len(solids))
    solids_varaition_dependence = compute_variation_dependence(solids)

    #--------------------------------------create solids_existence_dependence----------------------------
    #define existence dependence
    ifc_existence_dependence = {'a' : 'b',
                                'c' : 'd',
                                'e' : 'f',
                                'g' : 'h'}
    solids_existence_dependence = {}
    for i, s_i in enumerate(solids):
        if s_i.existence_dependence is False: continue
        for j, s_j in enumerate(solids):
            if s_i.existence_dependence == s_j.name:
                solids_existence_dependence[s_i] = s_j

    perfect_traj_solids = ifc_converter(data['IFC_PATH'])
    
    initial_beliefs = np.zeros(len(simulation_solids))
    for i, s in enumerate(simulation_solids):
        s_simulation_belief = s.schedule.cdf(data['current_time'])
        s.set_existance_belief_and_shader(s_simulation_belief)
        
        perfect_traj_solids[i] = s.clone()
        
        initial_beliefs[i] = s_simulation_belief

    simulation = RayCastingMap(simulation_solids)
    perfect_traj_simulation = RayCastingMap(perfect_traj_solids)

    #ESTIMATION INITALIZATION
    pose0 = data['ground_truth']['trajectory'][0]
    bounds_min, bounds_max, extent = simulation.bounds()
    N_particles = 10
    initial_particle_poses = np.vstack((np.random.normal(pose0[0], 0.2, N_particles),
                        np.random.normal(pose0[1], 0.2, N_particles),
                        np.full(N_particles,pose0[2]),
                        np.random.normal(pose0[3], np.radians(5.0), N_particles))).T
    simulated_sensor = data['sensor']
    simulated_sensor.piercing = True
    rbpf = RBPF(simulation, 
                simulated_sensor,
                initial_particle_poses,
                initial_beliefs,
                data['solids_existence_dependence'],
                data['solids_varaition_dependence'],
                data['U_COV'],
                reservoir_decay_rate = 0.2)

    rbpf_perfect = RBPF(perfect_traj_simulation, 
            simulated_sensor,
            np.array([pose0]),
            initial_beliefs,
            data['solids_existence_dependence'],
            data['solids_varaition_dependence'],
            data['U_COV'] * 1e-25,
            reservoir_decay_rate = 0.2)

    #DRAW
    #--------------INITIAL CONDITION AND PERFECT MAPPING------------
    drone = Drone(pose = pose0)
    drone.mount_sensor(data["sensor"])
    visApp = VisApp()
    [visApp.add_solid(s,"world") for s in perfect_traj_simulation.solids]
    visApp.redraw("world")
    visApp.setup_default_camera("world")
    visApp.add_solid(drone.solid, "world")
    vis_scan = ScanSolid("scan")
    visApp.add_solid(vis_scan, "world")
    trail_ground_truth = TrailSolid("trail_ground_truth", drone.pose[:3].reshape(1,3))
    visApp.add_solid(trail_ground_truth, "world")

    #----------------------METHOD-------------------------------
    visApp.add_scene("simulation","world")
    [visApp.add_solid(s,"simulation") for s in simulation.solids]
    visApp.redraw("simulation")
    visApp.setup_default_camera("simulation")
    sim_vis_particles = ParticlesSolid(poses = initial_particle_poses)
    visApp.add_solid(sim_vis_particles.lines, "simulation")
    visApp.add_solid(sim_vis_particles.tails, "simulation")
    sim_vis_trail_est = TrailSolid("trail_est", pose0[:3].reshape(1,3))
    visApp.add_solid(sim_vis_trail_est, "simulation")

    pose_mu, pose_cov = rbpf.get_expect_pose()
    expected_belief_map = rbpf.get_expected_belief_map()
    results_rbpf = {'pose_mu': [pose_mu],'pose_cov': [pose_cov],
                    'expected_belief_map': [expected_belief_map],
                    'perfect_traj_belief_map': [expected_belief_map]}

    #LOOP
    time.sleep(2)
    for t, (u,z) in enumerate(zip(data['measurements']['U'],data['measurements']['Z'])):
        #-----------------------------------estimate-------------------------------
        rbpf.step(u, z)
        pose_mu, pose_cov = rbpf.get_expect_pose()
        expected_belief_map = rbpf.get_expected_belief_map()

        #-----------------------------perfect trajectory---------------------------
        rbpf_perfect.particle_poses = np.array([data['ground_truth']['trajectory'][t+1]])
        rbpf_perfect.step(np.zeros(4), z)

        #-------------------------------store results------------------------------
        results_rbpf['pose_mu'].append(pose_mu)
        results_rbpf['pose_cov'].append(pose_cov)
        results_rbpf['expected_belief_map'].append(expected_belief_map)
        results_rbpf['perfect_traj_belief_map'].append(rbpf_perfect.particle_beliefs[0])

        #-----------------------------------draw-----------------------------------
        #update solids
        sim_vis_particles.update(rbpf.particle_poses, rbpf.weights)
        simulation.update_solids_beliefs(expected_belief_map)
        perfect_traj_simulation.update_solids_beliefs(rbpf_perfect.particle_beliefs[0])
        sim_vis_trail_est.update(pose_mu[:3].reshape(1,-1))
        drone.update_pose(rbpf_perfect.particle_poses[0])
        z_p = pose2z.transform_from(drone.pose, drone.sensor.scan_to_points(z))
        vis_scan.update(drone.pose[:3], z_p.T)
        trail_ground_truth.update(drone.pose[:3].reshape(1,-1))
        

        #updating visApp
        [visApp.update_solid(s,"world") for s in perfect_traj_simulation.solids]
        visApp.update_solid(vis_scan, "world")
        visApp.update_solid(drone.solid, "world")
        visApp.update_solid(trail_ground_truth, "world")
        visApp.update_solid(sim_vis_particles.lines, "simulation")
        visApp.update_solid(sim_vis_particles.tails, "simulation")
        [visApp.update_solid(s,"simulation") for s in simulation.solids]
        visApp.update_solid(sim_vis_trail_est,"simulation")
        visApp.redraw_all_scenes()

    #--------------------SAVE RESULTS--------------------
    results_rbpf["expected_belief_map"] = np.array(results_rbpf["expected_belief_map"])
    results_rbpf["pose_mu"] = np.array(results_rbpf["pose_mu"])
    results_rbpf["pose_cov"] = np.array(results_rbpf["pose_cov"])
    results_rbpf['perfect_traj_belief_map'] = np.array(results_rbpf['perfect_traj_belief_map'])

    results[rbpf_enum] = results_rbpf
    visApp.quit()

file = os.path.join(dir_path, f"25b_results.p")
pickle.dump(results, open(file, "wb"))
print('pickle dumped')