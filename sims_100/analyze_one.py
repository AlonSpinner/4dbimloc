import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from bim4loc.solids import ifc_converter, add_variations_from_yaml
from bim4loc.evaluation.evaluation import localiztion_error, map_entropy, cross_entropy_error

seednumber = 10
out_folder = "out_mid_noise"

dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path, out_folder ,f"data_{seednumber}.p")
data = pickle.Unpickler(open(file, "rb")).load()
file = os.path.join(dir_path, out_folder , f"results_{seednumber}.p")
results = pickle.Unpickler(open(file, "rb")).load()

colors = ['b', 'g', 'r', 'k', 'm']

#------------------------------------TRAJECTRY PLOTS--------------------------
fig, ax = plt.subplots()
ax.set_xlabel('Time [s]')
ax.set_ylabel('Error [m]')
ax.grid(True)
for i, res in enumerate(results.values()):
    traj_err, _ = localiztion_error(data['ground_truth']['trajectory'], \
                res['pose_mu'])
    ax.plot(traj_err, label = f'method {i}', color = colors[i], lw = 2)
ax.legend()
plt.show()

#------------------------------------CROSS ENTROPY PLOTS--------------------------
solids = ifc_converter(data['IFC_PATH'])
ground_truth_beliefs = np.zeros(len(solids),dtype = float)
for i, s in enumerate(solids):
    if s.name in data['ground_truth']['constructed_solids_names']:
        ground_truth_beliefs[i] = 1.0

fig, ax = plt.subplots()
ax.set_xlabel('Time [s]')
ax.set_ylabel('Entropy')
ax.set_ylim(0,15)
ax.grid(True)
for i, res in enumerate(results.values()):
    cross_entropy, cross_entropy_perfect_traj = cross_entropy_error(ground_truth_beliefs,
                                           res['expected_belief_map'],
                                           res['perfect_traj_belief_map'])
    ax.plot(cross_entropy, label = f'method {i}', color = colors[i], lw = 2)
    ax.plot(cross_entropy_perfect_traj, label = f'method {i} - perfect trajectory', color = colors[i], lw = 2, ls = '--')
ax.legend()
plt.show()

#------------------------------------SELF ENTROPY PLOTS--------------------------
solids = ifc_converter(data['IFC_PATH'])
ground_truth_beliefs = np.zeros(len(solids),dtype = float)
for i, s in enumerate(solids):
    if s.name in data['ground_truth']['constructed_solids_names']:
        ground_truth_beliefs[i] = 1.0

fig, ax = plt.subplots()
ax.set_xlabel('Time [s]')
ax.set_ylabel('Entropy')
ax.set_ylim(0,15)
ax.grid(True)
for i, res in enumerate(results.values()):
    self_entropy, self_entropy_perfect_traj = map_entropy(
                                    res['expected_belief_map'],
                                    res['perfect_traj_belief_map'])
    ax.plot(self_entropy, label = f'method {i}', color = colors[i], lw = 2)
    ax.plot(self_entropy_perfect_traj, label = f'method {i} - perfect trajectory', color = colors[i], lw = 2, ls = '--')
ax.legend()
plt.show()

#------------------------------------Electric Boxes PLOTS--------------------------
solids = ifc_converter(data['IFC_PATH'])
ground_truth_beliefs = np.zeros(len(solids),dtype = float)
for i, s in enumerate(solids):
    if s.name in data['ground_truth']['constructed_solids_names']:
        ground_truth_beliefs[i] = 1.0
gt_electric_boxes_names = [s.name for s in solids if s.ifc_type == 'IfcElectricDistributionBoard']
gt_electric_boxes_indicies = [i for i,s in enumerate(solids) if s.ifc_type == 'IfcElectricDistributionBoard']

yaml_file = os.path.join(dir_path, "complementry_IFC_data.yaml")
simulation_solids = ifc_converter(data['IFC_PATH'])
add_variations_from_yaml(simulation_solids, yaml_file)
sim_electric_boxes_indicies = [i for i,s in enumerate(simulation_solids) if s.ifc_type == 'IfcElectricDistributionBoard']
sim_electric_boxes_names = [s.vis_name for s in simulation_solids if s.ifc_type == 'IfcElectricDistributionBoard']

#for each gt box, check that it checks out, and that all variations are false
for k, res in enumerate(results.values()):
    N_boxes_got_right = np.zeros(len(results.keys()))
    for i,gt_box_name in enumerate(gt_electric_boxes_names):
        tick_box = True
        for j,sim_box_name in enumerate(sim_electric_boxes_names):
            if sim_box_name.startswith(gt_box_name):
                if sim_box_name.endswith(gt_box_name):
                    sim_belief = res['expected_belief_map'][-1][sim_electric_boxes_indicies[j]] > 0.9
                    gt_belief = ground_truth_beliefs[gt_electric_boxes_indicies[i]]
                    if sim_belief != gt_belief:
                        tick_box = False
                else:
                    sim_belief = res['expected_belief_map'][-1][sim_electric_boxes_indicies[j]] > 0.9
                    if sim_belief == 1:
                        tick_box = False
    if tick_box:
        N_boxes_got_right[k] += 1.0
N_boxes_got_right = N_boxes_got_right/len(gt_electric_boxes_names)
fig, ax = plt.subplots()
ax.bar(range(len(N_boxes_got_right)), N_boxes_got_right, color = colors)
plt.show()