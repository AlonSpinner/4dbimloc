import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from bim4loc.solids import ifc_converter
from bim4loc.evaluation.evaluation import localiztion_error, map_entropy, \
                        cross_entropy_error, belief_map_accuracy
from bim4loc.utils.load_yaml import load_parameters
from importlib import import_module
import bim4loc.binaries.paths as ifc_paths
plt.rcParams['font.size'] = '24'

def plots(out, seed_number):
    figs = {}

    out_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), out) 
    file = os.path.join(out_folder,"data",f"data_{seed_number}.p")
    data = pickle.Unpickler(open(file, "rb")).load()
    file = os.path.join(out_folder,"results",f"results_{seed_number}.p")
    results = pickle.Unpickler(open(file, "rb")).load()
    yaml_file = os.path.join(out_folder, "parameters.yaml")
    parameters_dict = load_parameters(yaml_file)
    ifc_file_path = getattr(import_module(ifc_paths.__name__),parameters_dict['IFC_PATH'])
    colors = ['b', 'g', 'r', 'k', 'm']

    #------------------------------------TRAJECTRY PLOTS--------------------------
    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Localization Error, m')
    ax.grid(True)
    for i, res in enumerate(results.values()):
        traj_err, _ = localiztion_error(data['ground_truth']['trajectory'], \
                    # np.array(res['best_pose']))
                    res['pose_mu'])
        ax.plot(traj_err, label = f'method {i}', color = colors[i], lw = 2)
    # ax.legend()
    plt.show()
    figs['localization_err']= fig

    #------------------------------------CROSS ENTROPY PLOTS--------------------------
    solids = ifc_converter(ifc_file_path)
    ground_truth_beliefs = np.zeros(len(solids),dtype = float)
    for i, s in enumerate(solids):
        if s.name in data['ground_truth']['constructed_solids_names']:
            ground_truth_beliefs[i] = 1.0

    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Belief Map Cross Entropy Error, bits')
    # ax.set_ylim(0,30)
    ax.grid(True)
    for i, res in enumerate(results.values()):
        cross_entropy, cross_entropy_perfect_traj = cross_entropy_error(ground_truth_beliefs,
                                            res['expected_belief_map'],
                                            res['perfect_traj_belief_map'])
        ax.plot(cross_entropy, label = f'method {i}', color = colors[i], lw = 2)
        ax.plot(cross_entropy_perfect_traj, label = f'method {i} - perfect trajectory', color = colors[i], lw = 2, ls = '--')
    # ax.legend()
    plt.show()
    figs['cross_entropy_error']= fig

    #------------------------------------SELF ENTROPY PLOTS--------------------------
    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Belief Map Self Entropy Error, bits')
    # ax.set_ylim(0,100)
    ax.grid(True)
    for i, res in enumerate(results.values()):
        self_entropy, self_entropy_perfect_traj = map_entropy(
                                        res['expected_belief_map'],
                                        res['perfect_traj_belief_map'])
        ax.plot(self_entropy, label = f'method {i}', color = colors[i], lw = 2)
        ax.plot(self_entropy_perfect_traj, label = f'method {i} - perfect trajectory', color = colors[i], lw = 2, ls = '--')
    # ax.legend()
    plt.show()
    figs['self_entropy_error']= fig

    # ------------------------------------Accuracy PLOTS--------------------------
    # fig = plt.figure(figsize = (16,8))
    # ax = fig.add_subplot(111)
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Belief Map Accuracy')
    # ax.set_ylim(0,1.1)
    # ax.grid(True)
    # for i, res in enumerate(results.values()):
    #     accuracy, perfect_accuracy = belief_map_accuracy(
    #                                     ground_truth_beliefs,
    #                                     res['expected_belief_map'],
    #                                     res['perfect_traj_belief_map'])
    #     ax.plot(accuracy, label = f'method {i}', color = colors[i], lw = 2)
    #     ax.plot(perfect_accuracy, label = f'method {i} - perfect trajectory', color = colors[i], lw = 2, ls = '--')
    # # ax.legend()
    # plt.show()
    return figs

if __name__ == "__main__":
    figs = plots("out7", 20)
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    full_folder = os.path.join(dir_path)
    for name, fig in figs.items():
        full_file = os.path.join(full_folder, f"{name}.png")    
        fig.savefig(full_file, dpi = 300, bbox_inches = 'tight')