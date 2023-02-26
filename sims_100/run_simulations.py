import numpy as np
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid, TrailSolid, ScanSolid, \
                            update_existence_dependence_from_yaml, add_variations_from_yaml, \
                            compute_variation_dependence_for_rbpf, compute_existence_dependece_for_rbpf
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.geometry import pose2z
import time
import pickle
import os
from bim4loc.rbpf.tracking.bimloc_robust import RBPF as robust
from bim4loc.rbpf.tracking.bimloc_semi_robust import RBPF as semi_robust
from bim4loc.rbpf.tracking.bimloc_simple import RBPF as simple
# from bim4loc.rbpf.tracking.bimloc_logodds_semi_robust import RBPF as logodds_semi_robust
from bim4loc.rbpf.tracking.bimloc_logodds import RBPF as logodds

def run_simulation(seed_number, data_folder ,out_folder, vis_on = False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    yaml_file = os.path.join(dir_path, "complementry_IFC_data.yaml")
    data_file = os.path.join(dir_path, data_folder , f"data_{seed_number}.p")
    data = pickle.Unpickler(open(data_file, "rb")).load()

    #SOME CONSTANTS
    pose0 = data['ground_truth']['trajectory'][0]
    N_particles = 10
    initial_particle_poses = np.random.multivariate_normal(pose0, data['U_COV'], N_particles)
    simulated_sensor = data['sensor']
    simulated_sensor.piercing = True
    simulated_sensor.std *= 2
    simulated_sensor.p0 = 0.4
    simulated_sensor.max_range_cutoff = False

    rbpf_methods = [robust, semi_robust, simple, logodds]
    results = {i : {} for i in range(1,len(rbpf_methods) + 1)}
    for (rbpf_enum, RBPF) in zip(results.keys(),rbpf_methods):

        #BUILD SIMULATION ENVIORMENT
        simulation_solids = ifc_converter(data['IFC_PATH'])
        add_variations_from_yaml(simulation_solids, yaml_file)
        update_existence_dependence_from_yaml(simulation_solids, yaml_file)

        #compute existence and variation dependence structures for rbpf
        solids_varaition_dependence = compute_variation_dependence_for_rbpf(simulation_solids)
        solids_existence_dependence = compute_existence_dependece_for_rbpf(simulation_solids)

        
        initial_beliefs = np.zeros(len(simulation_solids))
        for i, s in enumerate(simulation_solids):
            s_simulation_belief = s.schedule.cdf(data['current_time'])
            s.set_existance_belief_and_shader(s_simulation_belief)
            initial_beliefs[i] = s_simulation_belief

        perfect_traj_solids = [s.clone() for s in simulation_solids]

        simulation = RayCastingMap(simulation_solids)
        perfect_traj_simulation = RayCastingMap(perfect_traj_solids)

        #ESTIMATION INITALIZATION        
        rbpf = RBPF(simulation, 
                    simulated_sensor,
                    initial_particle_poses.copy(),
                    initial_beliefs.copy(),
                    solids_existence_dependence,
                    solids_varaition_dependence,
                    data['U_COV'],
                    max_steps_to_resample = 10,
                    reservoir_decay_rate = 0.2)

        rbpf_perfect = RBPF(perfect_traj_simulation, 
                simulated_sensor,
                np.array([pose0.copy()]),
                initial_beliefs,
                solids_existence_dependence,
                solids_varaition_dependence,
                data['U_COV'] * 1e-25)

        #DRAW
        if vis_on:
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
                        'particle_poses': [initial_particle_poses],
                        'particle_weights': [np.ones(N_particles)/N_particles],
                        'particle_beliefs': [initial_beliefs],
                        'perfect_traj_belief_map': [expected_belief_map]}

        np.random.seed(seed_number)
        
        #LOOP
        time.sleep(2)
        for t, (u,z,z_perfect) in enumerate(zip(data['measurements']['U'],
                                                data['measurements']['Z'],
                                                data['measurements']['Z_perfect'])):

            #-----------------------------------estimate-------------------------------
            rbpf.step(u, z)
            pose_mu, pose_cov = rbpf.get_expect_pose()
            expected_belief_map = rbpf.get_expected_belief_map()

            if t == len(data['measurements']['U']) or t == 0:
                rbpf.resample()

            #-----------------------------perfect trajectory---------------------------
            rbpf_perfect.particle_poses = np.array([data['ground_truth']['trajectory'][t+1]])
            rbpf_perfect.step(np.zeros(4), z_perfect)

            #-------------------------------store results------------------------------
            results_rbpf['pose_mu'].append(pose_mu)
            results_rbpf['pose_cov'].append(pose_cov)
            results_rbpf['expected_belief_map'].append(expected_belief_map)
            results_rbpf['perfect_traj_belief_map'].append(rbpf_perfect.particle_beliefs[0].copy())
            results_rbpf['particle_poses'].append(rbpf.particle_poses.copy())
            results_rbpf['particle_weights'].append(rbpf.weights.copy())
            results_rbpf['particle_beliefs'].append(rbpf.particle_beliefs.copy())

            #-----------------------------------draw-----------------------------------
            if vis_on:
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
        if vis_on:
            visApp.quit()

    file = os.path.join(dir_path, out_folder, f"results_{seed_number}.p")
    pickle.dump(results, open(file, "wb"))
    print('pickle dumped')