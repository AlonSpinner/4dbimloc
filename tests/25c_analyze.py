import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from bim4loc.solids import ifc_converter
from bim4loc.evaluation.evaluation import localiztion_error, map_entropy, \
                        cross_entropy_error, belief_map_accuracy

dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path, "25a_data.p")
data = pickle.Unpickler(open(file, "rb")).load()
file = os.path.join(dir_path, "25b_results.p")
results = pickle.Unpickler(open(file, "rb")).load()

colors = ['b', 'g', 'r', 'k', 'm']

plt.rcParams['font.size'] = '24'
# data['IFC_PATH'] = '/home/alon18/repos/4dbimloc/bim4loc/binaries/arena.ifc'

#------------------------------------TRAJECTRY PLOTS--------------------------
fig = plt.figure(figsize = (16,8))
ax = fig.add_subplot(111)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Trajectory Error [m]')
ax.grid(True)
for i, res in enumerate(results.values()):
    traj_err, _ = localiztion_error(data['ground_truth']['trajectory'], \
                res['pose_mu'])
    ax.plot(traj_err, label = f'method {i}', color = colors[i], lw = 2)
# ax.legend()
plt.show()

#------------------------------------CROSS ENTROPY PLOTS--------------------------
solids = ifc_converter(data['IFC_PATH'])
ground_truth_beliefs = np.zeros(len(solids),dtype = float)
for i, s in enumerate(solids):
    if s.name in data['ground_truth']['constructed_solids_names']:
        ground_truth_beliefs[i] = 1.0

fig = plt.figure(figsize = (16,8))
ax = fig.add_subplot(111)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Belief Map Cross Entropy')
ax.set_ylim(0,15)
ax.grid(True)
for i, res in enumerate(results.values()):
    cross_entropy, cross_entropy_perfect_traj = cross_entropy_error(ground_truth_beliefs,
                                           res['expected_belief_map'],
                                           res['perfect_traj_belief_map'])
    ax.plot(cross_entropy, label = f'method {i}', color = colors[i], lw = 2)
    ax.plot(cross_entropy_perfect_traj, label = f'method {i} - perfect trajectory', color = colors[i], lw = 2, ls = '--')
# ax.legend()
plt.show()

#------------------------------------SELF ENTROPY PLOTS--------------------------
solids = ifc_converter(data['IFC_PATH'])
ground_truth_beliefs = np.zeros(len(solids),dtype = float)
for i, s in enumerate(solids):
    if s.name in data['ground_truth']['constructed_solids_names']:
        ground_truth_beliefs[i] = 1.0

fig = plt.figure(figsize = (16,8))
ax = fig.add_subplot(111)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Belief Map Self Entropy')
ax.set_ylim(0,15)
ax.grid(True)
for i, res in enumerate(results.values()):
    self_entropy, self_entropy_perfect_traj = map_entropy(
                                    res['expected_belief_map'],
                                    res['perfect_traj_belief_map'])
    ax.plot(self_entropy, label = f'method {i}', color = colors[i], lw = 2)
    ax.plot(self_entropy_perfect_traj, label = f'method {i} - perfect trajectory', color = colors[i], lw = 2, ls = '--')
# ax.legend()
plt.show()

#------------------------------------Accuracy PLOTS--------------------------
fig = plt.figure(figsize = (16,8))
ax = fig.add_subplot(111)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Belief Map Accuracy')
ax.set_ylim(0,1)
ax.grid(True)
for i, res in enumerate(results.values()):
    accuracy, perfect_accuracy = belief_map_accuracy(
                                    ground_truth_beliefs,
                                    res['expected_belief_map'],
                                    res['perfect_traj_belief_map'])
    ax.plot(accuracy, label = f'method {i}', color = colors[i], lw = 2)
    ax.plot(perfect_accuracy, label = f'method {i} - perfect trajectory', color = colors[i], lw = 2, ls = '--')
# ax.legend()
plt.show()