import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['font.size'] = '24'
dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path ,"analyzed_data.p")
d = pickle.Unpickler(open(file, "rb")).load()

colors = ['b', 'g', 'r', 'k']
#-----------------------------------Failures --------------------------------------------
# fig = plt.figure(figsize = (10,8))
# ax = fig.add_subplot(111)
# ax_f = ax.bar(np.array(list(d['N_traj_failures'].keys())),
#        d['N_failures'].values(),
#        edgecolor = 'black',
#        align='center', alpha=0.5,
#        color = colors)
# ax.set_xticks(np.array(list(d['N_failures'].keys())))
# ax.set_ylabel('Failures')
# ax.set_xticklabels(['BPFS', 'BPFS-t', 'BPFS-tg', 'log-odds'])
# plt.show()

#----------------------------------- Mean Trajectory Error ------------------------------------
fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)

for i, m in enumerate(d['mean_traj_err'].values()):
    ax.boxplot(m, positions = [i+1],
     showfliers = True, 
     medianprops=dict(linewidth=3.0, color='k'),
     patch_artist = True, boxprops = dict(facecolor = colors[i], alpha = 0.5, linewidth = 2.0),
     flierprops = dict(markerfacecolor = colors[i], marker = 'o', markersize = 7.0, markeredgecolor = 'k', alpha = 0.5))
ax.set_ylabel('Mean Trajectory Error, m')
ax.set_xticks(np.array(list(d['mean_traj_err'].keys())))
ax.set_xticklabels(['BPFS', 'BPFS-t', 'BPFS-tg', 'log-odds'])
ax.grid(True)
plt.show()
#----------------------------------- Mean Cross Entropy Error ------------------------------------
fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)
for i, m in enumerate(d['final_mean_ce_err'].values()):
    ax.boxplot(m, positions = [i+1],
     showfliers = True, 
     medianprops=dict(linewidth=3.0, color='k'),
     patch_artist = True, boxprops = dict(facecolor = colors[i], alpha = 0.5, linewidth = 2.0),
     flierprops = dict(markerfacecolor = colors[i], marker = 'o', markersize = 7.0, markeredgecolor = 'k', alpha = 0.5))
ax.set_ylabel('Mean Cross Entropy Error At Terminal, bits')
ax.set_xticks(np.array(list(d['final_mean_ce_err'].keys())))
ax.set_xticklabels(['BPFS', 'BPFS-t', 'BPFS-tg', 'log-odds'])
ax.grid(True)
plt.show()
#----------------------------------- Electric Boxes ------------------------------------
method_p_boxes = {}
for method_i in d['analyzed_by_method'].keys():
    p_seeds = []
    for seed in range(len(d['analyzed_by_method'][method_i])):#d['sucesses'][method_i]:
        p_boxes = d['analyzed_by_method'][method_i][seed]['percentile_boxes']
        if not(np.isnan(p_boxes)):
            p_seeds.append(p_boxes)
    method_p_boxes[method_i] = np.array(p_seeds)

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)
for i, m in enumerate(method_p_boxes.values()):
    ax.boxplot(100 * m, positions = [i+1],
     showfliers = True, 
     medianprops=dict(linewidth=3.0, color='k'),
     patch_artist = True, boxprops = dict(facecolor = colors[i], alpha = 0.5, linewidth = 2.0),
     flierprops = dict(markerfacecolor = colors[i], marker = 'o', markersize = 7.0, markeredgecolor = 'k', alpha = 0.5))
ax.set_ylabel('Wall-Hung Electrical Boxes Accuracy at Terminal, %', fontdict={'fontsize': 20})
ax.set_xticks(np.array(list(method_p_boxes.keys())))
ax.set_xticklabels(['BPFS', 'BPFS-t', 'BPFS-tg', 'log-odds'])
ax.grid(True)
plt.show()

#----------------------------------- Accuracy ------------------------------------
fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
for i, m in enumerate(d['final_acc'].values()):
    ax.boxplot(100 * np.array(m), positions = [i+1],
     showfliers = True, 
     medianprops=dict(linewidth=3.0, color='k'),
     patch_artist = True, boxprops = dict(facecolor = colors[i], alpha = 0.5, linewidth = 2.0),
     flierprops = dict(markerfacecolor = colors[i], marker = 'o', markersize = 7.0, markeredgecolor = 'k', alpha = 0.5))
ax.set_ylabel('Belief Map Accuracy at Terminal, %')
ax.set_xticks(np.array(list(d['final_acc'].keys())))
ax.set_xticklabels(['BPFS', 'BPFS-t', 'BPFS-tg', 'log-odds'])
ax.grid(True)
plt.show()