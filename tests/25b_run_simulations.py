import numpy as np
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid, TrailSolid
from bim4loc.maps import RayCastingMap
import time
import logging
import pickle
import os
from bim4loc.rbpf.tracking.bimloc import RBPF as RBPF_0
from bim4loc.rbpf.tracking.bimloc_partial import RBPF as RBPF_1
from bim4loc.rbpf.tracking.bimloc_logodds import RBPF as RBPF_2

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path, "25a_data.p")
data = pickle.Unpickler(open(file, "rb")).load()

solids = ifc_converter(data['IFC_PATH'])

constructed_solids = []
for s in solids:
    if s.name in data['constructed_solids_names']:
        constructed_solids.append(s.clone())

results = {0: {}, 1: {}, 2: {}}
for rbpf_enum, RBPF in enumerate([RBPF_0, RBPF_1, RBPF_2]):

    #BUILD SIMULATION ENVIORMENT
    solids = ifc_converter(data['IFC_PATH'])

    initial_beliefs = np.zeros(len(solids))
    for i, s in enumerate(solids):
        s_simulation_belief = s.schedule.cdf(data['current_time'])
        s.set_existance_belief_and_shader(s_simulation_belief)
        
        initial_beliefs[i] = s_simulation_belief

    simulation = RayCastingMap(solids)

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

    dead_reckoning = pose0

    #DRAW
    #--------------INITIAL CONDITION AND DEAD RECKONING------------
    visApp = VisApp()
    [visApp.add_solid(s,"world") for s in constructed_solids]
    visApp.redraw("world")
    visApp.setup_default_camera("world")

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

    results_rbpf = {'pose_mu': [],'pose_cov': [],'expected_belief_map': []}

    #LOOP
    time.sleep(2)
    for t, (u,z) in enumerate(zip(data['measurements']['U'],data['measurements']['Z'])):
        #-----------------------------------estimate-------------------------------
        rbpf.step(u, z)
        pose_mu, pose_cov = rbpf.get_expect_pose()
        expected_belief_map = rbpf.get_expected_belief_map()

        #-------------------------------store results------------------------------
        results_rbpf['pose_mu'].append(pose_mu)
        results_rbpf['pose_cov'].append(pose_cov)
        results_rbpf['expected_belief_map'].append(expected_belief_map)

        #-----------------------------------draw-----------------------------------
        #update solids
        sim_vis_particles.update(rbpf.particle_poses, rbpf.weights)
        simulation.update_solids_beliefs(expected_belief_map)
        sim_vis_trail_est.update(pose_mu[:3].reshape(1,-1))

        #updating visApp
        visApp.update_solid(sim_vis_particles.lines, "simulation")
        visApp.update_solid(sim_vis_particles.tails, "simulation")
        [visApp.update_solid(s,"simulation") for s in simulation.solids]
        visApp.update_solid(sim_vis_trail_est,"simulation")
        visApp.redraw_all_scenes()
        
    #--------------------SAVE RESULTS--------------------
    results_rbpf["expected_belief_map"] = np.array(results_rbpf["expected_belief_map"])
    results_rbpf["pose_mu"] = np.array(results_rbpf["pose_mu"])
    results_rbpf["pose_cov"] = np.array(results_rbpf["pose_cov"])

    results[rbpf_enum] = results_rbpf
    visApp.quit()

file = os.path.join(dir_path, f"25b_results.p")
pickle.dump(results, open(file, "wb"))
print('pickle dumped')