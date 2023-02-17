import numpy as np
import pickle
import os
from bim4loc.solids import ifc_converter, add_variations_from_yaml
from bim4loc.evaluation.evaluation import localiztion_error, map_entropy, \
    cross_entropy_error, percentile_boxes_right, belief_map_accuracy, maps_average_distance

out_folder = "out_mid_noise"

data_by_seed = []
results_by_seed = []
analyzed_by_seed = []
ground_truth_maps = []
max_seed = 19
for seednumber in range(max_seed):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(dir_path, out_folder ,f"data_{seednumber}.p")
    data = pickle.Unpickler(open(file, "rb")).load()
    data['IFC_PATH'] = '/home/alon18/repos/4dbimloc/bim4loc/binaries/arena.ifc'

    file = os.path.join(dir_path, out_folder , f"results_{seednumber}.p")
    results = pickle.Unpickler(open(file, "rb")).load()

    solids = ifc_converter(data['IFC_PATH'])
    ground_truth_beliefs = np.zeros(len(solids),dtype = float)
    for i, s in enumerate(solids):
        if s.name in data['ground_truth']['constructed_solids_names']:
            ground_truth_beliefs[i] = 1.0
    ground_truth_maps.append(ground_truth_beliefs)

    gt_electric_boxes_names = [s.vis_name for s in solids if s.ifc_type == 'IfcElectricDistributionBoard']
    gt_electric_boxes_indicies = [i for i,s in enumerate(solids) if s.ifc_type == 'IfcElectricDistributionBoard']

    yaml_file = os.path.join(dir_path, "complementry_IFC_data.yaml")
    simulation_solids = ifc_converter(data['IFC_PATH'])
    add_variations_from_yaml(simulation_solids, yaml_file)
    sim_electric_boxes_indicies = [i for i,s in enumerate(simulation_solids) if s.ifc_type == 'IfcElectricDistributionBoard']
    sim_electric_boxes_names = [s.vis_name for s in simulation_solids if s.ifc_type == 'IfcElectricDistributionBoard']

    analyzed = {}
    for i, res in enumerate(results.values()):
        traj_err, _ = localiztion_error(data['ground_truth']['trajectory'], \
                    res['pose_mu'])
        cross_entropy, cross_entropy_perfect_traj = cross_entropy_error(ground_truth_beliefs,
                                        res['expected_belief_map'],
                                        res['perfect_traj_belief_map'])
        self_entropy, self_entropy_perfect_traj = map_entropy(
                                    res['expected_belief_map'],
                                    res['perfect_traj_belief_map'])

        accuracy, accuracy_perfect_traj = belief_map_accuracy(ground_truth_beliefs,
                                    res['expected_belief_map'],
                                    res['perfect_traj_belief_map'])

        p_boxes = percentile_boxes_right(res['expected_belief_map'],ground_truth_beliefs,
                                            gt_electric_boxes_names, gt_electric_boxes_indicies,
                                            sim_electric_boxes_indicies, sim_electric_boxes_names,
                                            data["measurements"]["electric_boxes_seen_counter"])

        analyzed[i+1] = {'traj_err': traj_err,
                    'cross_entropy': cross_entropy,
                    'cross_entropy_perfect_traj': cross_entropy_perfect_traj,
                    'self_entropy': self_entropy,
                    'self_entropy_perfect_traj': self_entropy_perfect_traj,
                    'percentile_boxes' : p_boxes,
                    'accuracy': accuracy,
                    'accuracy_perfect_traj': accuracy_perfect_traj}

    analyzed_by_seed.append(analyzed)
    results_by_seed.append(results)
    data_by_seed.append(data)

def by_seed_to_by_method(analyzed_list : dict):
    methods = list(analyzed_list[0].keys())
    by_method = {method_i: [] for method_i in methods}
    for seed in range(len(analyzed_list)):
        analyzed = analyzed_list[seed]
        for method_i in analyzed.keys():
            by_method[method_i].append(analyzed[method_i])
    return by_method

def failure_by_traj_error(analyzed_method):
    fail_seeds = []
    for seed in range(len(analyzed_method)):
        if np.any(analyzed_method[seed]['traj_err'] > 1.0):
            fail_seeds.append(seed)
    return fail_seeds

def failure_by_cross_entropy(analyzed_method):
    fail_seeds = []
    for seed in range(len(analyzed_method)):
        if analyzed_method[seed]['cross_entropy'][-1] > analyzed_method[seed]['cross_entropy'][0]:
            fail_seeds.append(seed)
    return fail_seeds

def average_traj_err(analyzed_method):
    #return mean and std
    mu = np.mean(analyzed_method['traj_err'])
    std = np.std(analyzed_method['traj_err'])
    return mu, std 

def average_cross_entropy(analyzed_method):
    #return mean and std
    mu = np.mean(analyzed_method['cross_entropy'][-1])
    std = np.std(analyzed_method['cross_entropy'][-1])
    return mu, std

def final_accuracy(analyzed_method):
    return analyzed_method['accuracy'][-1]
    
analyzed_by_method = by_seed_to_by_method(analyzed_by_seed)

traj_failures = {}
N_traj_failures = {}
cross_entropy_failures = {}
N_cross_entropy_failures = {}
failures = {}
N_failures = {}
for method_i in analyzed_by_method.keys():
    traj_failures[method_i] = failure_by_traj_error(analyzed_by_method[method_i])
    N_traj_failures[method_i] = len(traj_failures[method_i])
    cross_entropy_failures[method_i] = failure_by_cross_entropy(analyzed_by_method[method_i])
    N_cross_entropy_failures[method_i] = len(cross_entropy_failures[method_i])

    failures[method_i] = list(set(traj_failures[method_i] + cross_entropy_failures[method_i]))
    N_failures[method_i] = len(failures[method_i])

sucesses = {} ; N_sucesses = {}
mean_traj_err = {}
final_mean_ce_err = {}
final_acc = {}
for method_i in analyzed_by_method.keys():
    sucesses[method_i] = list(set(range(max_seed)) - set(failures[method_i]))
    N_sucesses[method_i] = len(sucesses[method_i])
    
    traj_mu_seeds = []
    final_ce_mu_seeds = []
    final_acc_seeds = []
    for seed in range(max_seed):
        final_ce_mu_seed, _ = average_cross_entropy(analyzed_by_method[method_i][seed])
        final_ce_mu_seeds.append(final_ce_mu_seed)

        traj_mu_seed, _ = average_traj_err(analyzed_by_method[method_i][seed])
        traj_mu_seeds.append(traj_mu_seed)

        final_acc_seeds.append(final_accuracy(analyzed_by_method[method_i][seed]))

    final_mean_ce_err[method_i] = final_ce_mu_seeds
    mean_traj_err[method_i] = traj_mu_seeds
    final_acc[method_i] = final_acc_seeds


ground_truth_maps = np.array(ground_truth_maps)
hamming_map_avg_dist, hamming_min_dist_by_map = maps_average_distance(ground_truth_maps, "hamming")
map_length = ground_truth_maps.shape[1]
normalized_hamming_map_avg_dist = hamming_map_avg_dist/map_length
normalized_hamming_min_dist_by_map = hamming_min_dist_by_map/map_length

jaccard_map_avg_dist, jaccard_min_dist_by_map = maps_average_distance(ground_truth_maps, "jaccard")

analyzed_data = {'analyzed_by_method': analyzed_by_method,
                'traj_failures': traj_failures,
                'N_traj_failures': N_traj_failures,
                'cross_entropy_failures': cross_entropy_failures,
                'N_cross_entropy_failures': N_cross_entropy_failures,
                'failures': failures,
                'N_failures': N_failures,
                'sucesses': sucesses,
                'N_sucesses': N_sucesses,
                'mean_traj_err': mean_traj_err,
                'final_mean_ce_err': final_mean_ce_err,
                'final_acc': final_acc,
                'gt_maps_hamming': {'norm_avg_dist': normalized_hamming_map_avg_dist,
                                    'norm_min_dist_by_map': normalized_hamming_min_dist_by_map},
                'gt_maps_jaccard': {'avg_dist': jaccard_map_avg_dist,
                                    'min_dist_by_map': jaccard_min_dist_by_map}
                }

file = os.path.join(dir_path, "analyzed_data.p")
pickle.dump(analyzed_data, open(file, "wb"))
print('pickle dumped')