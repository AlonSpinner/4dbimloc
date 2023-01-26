import numpy as np
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid, TrailSolid
from bim4loc.maps import RayCastingMap
import time
import logging
import pickle
import os
import matplotlib.pyplot as plt
from bim4loc.evaluation.evaluation import localiztion_error, map_entropy, cross_entropy_error

dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path, "25a_data.p")
data = pickle.Unpickler(open(file, "rb")).load()
file = os.path.join(dir_path, "25b_results.p")
results = pickle.Unpickler(open(file, "rb")).load()

#------------------------------------TRAJECTRY PLOTS--------------------------
rbpf_0_traj_err, _ = localiztion_error(data['ground_truth']['trajectory'], \
                results[0]['pose_mu'])
rbpf_1_traj_err, _ = localiztion_error(data['ground_truth']['trajectory'], \
                results[1]['pose_mu'])
rbpf_2_traj_err, _ = localiztion_error(data['ground_truth']['trajectory'], \
                results[2]['pose_mu'])
fig, ax = plt.subplots()
ax.plot(rbpf_0_traj_err, label = 'method 1', color = 'b', lw = 2)
ax.plot(rbpf_1_traj_err, label = 'method 2', color = 'g', lw = 2)
ax.plot(rbpf_2_traj_err, label = 'method 3', color = 'r', lw = 2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Error [m]')
ax.grid(True)
ax.legend()
plt.show()

#------------------------------------CROSS ENTROPY PLOTS--------------------------
solids = ifc_converter(data['IFC_PATH'])
ground_truth_beliefs = np.zeros(len(solids),dtype = float)
for i, s in enumerate(solids):
    if s.name in data['ground_truth']['constructed_solids_names']:
        ground_truth_beliefs[i] = 1.0

rbpf_0_cross_entropy, rbpf_0_perfect_cross_entropy = cross_entropy_error(ground_truth_beliefs,
                                           results[0]['expected_belief_map'],
                                           results[0]['perfect_traj_belief_map'])
rbpf_1_cross_entropy, rbpf_1_perfect_cross_entropy = cross_entropy_error(ground_truth_beliefs,
                                           results[1]['expected_belief_map'],
                                           results[1]['perfect_traj_belief_map'])
rbpf_2_cross_entropy, rbpf_2_perfect_cross_entropy = cross_entropy_error(ground_truth_beliefs,
                                           results[2]['expected_belief_map'],
                                           results[2]['perfect_traj_belief_map'])

fig, ax = plt.subplots()
ax.plot(rbpf_0_cross_entropy, label = 'method 1', color = 'b', lw = 2)
ax.plot(rbpf_0_perfect_cross_entropy, label = 'method 1 - perfect trajectory', color = 'b', lw = 2, ls = '--')
ax.plot(rbpf_1_cross_entropy, label = 'method 2', color = 'g', lw = 2)
ax.plot(rbpf_1_perfect_cross_entropy, label = 'method 2 - perfect trajectory', color = 'g', lw = 2, ls = '--')
ax.plot(rbpf_2_cross_entropy, label = 'method 3', color = 'r', lw = 2)
ax.plot(rbpf_2_perfect_cross_entropy, label = 'method 3 - perfect trajectory', color = 'r', lw = 2, ls = '--')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Entropy')
ax.grid(True)
ax.legend()
plt.show()

#------------------------------------SELF ENTROPY PLOTS--------------------------
solids = ifc_converter(data['IFC_PATH'])
ground_truth_beliefs = np.zeros(len(solids),dtype = float)
for i, s in enumerate(solids):
    if s.name in data['ground_truth']['constructed_solids_names']:
        ground_truth_beliefs[i] = 1.0

rbpf_0_map_entropy, rbpf_0_perfect_map_entropy = map_entropy(
                                    results[0]['expected_belief_map'],
                                    results[0]['perfect_traj_belief_map'])
rbpf_1_map_entropy, rbpf_1_perfect_map_entropy = map_entropy(
                                    results[1]['expected_belief_map'],
                                    results[1]['perfect_traj_belief_map'])
rbpf_2_map_entropy, rbpf_2_perfect_map_entropy = map_entropy(
                                    results[2]['expected_belief_map'],
                                    results[2]['perfect_traj_belief_map'])

fig, ax = plt.subplots()
ax.plot(rbpf_0_map_entropy, label = 'method 1', color = 'b', lw = 2)
ax.plot(rbpf_0_perfect_map_entropy, label = 'method 1 - perfect trajectory', color = 'b', lw = 2, ls = '--')
ax.plot(rbpf_1_map_entropy, label = 'method 2', color = 'g', lw = 2)
ax.plot(rbpf_1_perfect_map_entropy, label = 'method 2 - perfect trajectory', color = 'g', lw = 2, ls = '--')
ax.plot(rbpf_2_map_entropy, label = 'method 3', color = 'r', lw = 2)
ax.plot(rbpf_2_perfect_map_entropy, label = 'method 3 - perfect trajectory', color = 'r', lw = 2, ls = '--')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Entropy')
ax.grid(True)
ax.legend()
plt.show()                             