import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from bim4loc.solids import ifc_converter
from bim4loc.evaluation.evaluation import localiztion_error, map_entropy, cross_entropy_error

out_folder = "out_large_noise"

data_by_seed = []
results_by_seed = []
analyzed_by_seed = []
max_seed = 20
for seednumber in range(max_seed):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(dir_path, out_folder ,f"data_{seednumber}.p")
    data = pickle.Unpickler(open(file, "rb")).load()

    file = os.path.join(dir_path, out_folder , f"results_{seednumber}.p")
    results = pickle.Unpickler(open(file, "rb")).load()

    solids = ifc_converter(data['IFC_PATH'])
    ground_truth_beliefs = np.zeros(len(solids),dtype = float)
    for i, s in enumerate(solids):
        if s.name in data['ground_truth']['constructed_solids_names']:
            ground_truth_beliefs[i] = 1.0

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

        analyzed[i+1] = {'traj_err': traj_err,
                    'cross_entropy': cross_entropy,
                    'cross_entropy_perfect_traj': cross_entropy_perfect_traj,
                    'self_entropy': self_entropy,
                    'self_entropy_perfect_traj': self_entropy_perfect_traj}
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
        if np.any(analyzed_method[seed]['cross_entropy'][10:] > analyzed_method[seed]['cross_entropy'][0]):
            fail_seeds.append(seed)
    return fail_seeds

def average_traj_err(analyzed_method):
    #return mean and std
    mu = np.mean(analyzed_method['traj_err'])
    std = np.std(analyzed_method['traj_err'])
    return mu, std 
    
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

sucesses = {}
N_sucesses = {}
mean_traj_err = {}
std_traj_err = {}
for method_i in analyzed_by_method.keys():
    sucesses[method_i] = list(set(range(max_seed)) - set(failures[method_i]))
    N_sucesses[method_i] = len(sucesses[method_i])
    
    mu_seeds = []
    std_seeds = []
    for seed in sucesses[method_i]:
        mu_seed, std_seed = average_traj_err(analyzed_by_method[method_i][seed])
        mu_seeds.append(mu_seed)
        std_seeds.append(std_seed)
    mean_traj_err[method_i] = (np.mean(mu_seeds),np.std(mu_seeds))
    std_traj_err[method_i] = (np.mean(std_seeds),np.std(std_seeds))

colors = ['b', 'g', 'r', 'k']
#-----------------------------------Failures --------------------------------------------
fig, ax = plt.subplots()
ax.set_xlabel('Method ')
ax.set_ylabel('Failures')
width = 0.25
ax_cef = ax.bar(np.array(list(N_cross_entropy_failures.keys())) - width,
       N_cross_entropy_failures.values(),
       edgecolor = 'black',
       align='center', alpha=0.5,
       color = colors,
       width = width,
       tick_label = [str(key) for key in mean_traj_err.keys()])
ax_f = ax.bar(np.array(list(N_traj_failures.keys())) + width,
       N_failures.values(),
       edgecolor = 'black',
       align='center', alpha=1.0,
       color = colors,
       width = width,
       tick_label = [str(key) for key in mean_traj_err.keys()])
ax_tf = ax.bar(np.array(list(N_traj_failures.keys())),
       N_traj_failures.values(),
       edgecolor = 'black',
       align='center', alpha=0.7,
       color = colors,
       width = width,
       tick_label = [str(key) for key in mean_traj_err.keys()])
plt.show()

#----------------------------------- Mean Trajectory Error ------------------------------------
fig, ax = plt.subplots()
ax.set_xlabel('Method ')
ax.set_ylabel('Error [m]')
ax.grid(False)
width = 0.25
ax_mean = ax.bar(np.array(list(mean_traj_err.keys())) - width/2,
       [max(0.0,m[0]) for m in mean_traj_err.values()],
       yerr=[max(0.0,m[1]) for m in mean_traj_err.values()],
       align='center', alpha=0.5,
       ecolor='black', capsize=10,
       width = width,
       color = colors,
       tick_label = [str(key) for key in mean_traj_err.keys()])

ax_std = ax.bar(np.array(list(std_traj_err.keys())) + width/2,
       [max(0.0,s[0]) for s in std_traj_err.values()],
       yerr=[max(0.0,s[1]) for s in std_traj_err.values()],
       align='center', alpha=0.5,
       ecolor='black', capsize=10,
       width = width,
       color = colors,
       tick_label = [str(key) for key in std_traj_err.keys()])
plt.show()



