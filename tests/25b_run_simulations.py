import numpy as np
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid, TrailSolid, ArrowSolid
from bim4loc.maps import RayCastingMap
from bim4loc.geometry.pose2z import compose_s
import time
import logging
import pickle
import os
from bim4loc.rbpf.tracking.exact import RBPF

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path, "25a_data.p")
data = pickle.Unpickler(open(file, "rb")).load()

#BUILD SIMULATION
simulation_solids = ifc_converter(data['IFC_PATH'])
initial_beliefs = np.zeros(len(simulation_solids))
for i, s in enumerate(simulation_solids):
    s_simulation_belief = s.schedule.cdf(data['current_time'])
    s.set_existance_belief_and_shader(s_simulation_belief)
    
    initial_beliefs[i] = s_simulation_belief

simulation = RayCastingMap(simulation_solids)
m1_simulation = RayCastingMap([s.clone() for s in simulation_solids])

#SPREAD PARTICLES
pose0 = data['history']['gt_traj'][0]
bounds_min, bounds_max, extent = simulation.bounds()
N_particles = 10
initial_particle_poses = np.vstack((np.random.normal(pose0[0], 0.2, N_particles),
                       np.random.normal(pose0[1], 0.2, N_particles),
                       np.full(N_particles,pose0[2]),
                       np.random.normal(pose0[3], np.radians(5.0), N_particles))).T

#DRAW
#----------------------INITIAL CONDITION-------------------
visApp = VisApp()
[visApp.add_solid(s,"world") for s in simulation.solids]
visApp.redraw("world")
visApp.setup_default_camera("world")
dead_reck_arrow = ArrowSolid("dead_reck_arrow", 1.0, pose0)
visApp.add_solid(dead_reck_arrow, "world")
dead_reck_vis_trail_est = TrailSolid("trail_est", pose0[:3].reshape(1,3))
visApp.add_solid(dead_reck_vis_trail_est, "world")

#----------------------METHOD1-------------------------------
visApp.add_window("estimation")
visApp.add_scene("method1","estimation")
[visApp.add_solid(s,"method1") for s in simulation.solids]
visApp.redraw("method1")
visApp.setup_default_camera("method1")
m1_vis_particles = ParticlesSolid(poses = initial_particle_poses)
visApp.add_solid(m1_vis_particles.lines, "method1")
visApp.add_solid(m1_vis_particles.tails, "method1")
m1_vis_trail_est = TrailSolid("trail_est", pose0[:3].reshape(1,3))
visApp.add_solid(m1_vis_trail_est, "method1")

simulated_sensor = data['sensor']
simulated_sensor.piercing = True
rbpf = RBPF(simulation, simulated_sensor, resample_rate = 4, U_COV = data['U_COV'])

#LOOP
time.sleep(2)
for t, (u,z) in enumerate(zip(data['history']['U'],data['history']['Z'])):

    dead_reckoning = compose_s(dead_reckoning, u)

    #move drone
    pass
    #updating drawings
    # visApp.update_solid(vis_scan)
    # visApp.update_solid(drone.solid)
    # trail_ground_truth.update(drone.pose[:3].reshape(1,-1))
    # visApp.update_solid(trail_ground_truth, "world")
    # visApp.redraw_all_scenes()