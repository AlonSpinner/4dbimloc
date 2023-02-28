import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['font.size'] = '24'
dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path, "statistical_analysis" ,"statistical_data.p")
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
method_boxes = {}
for method_i in d['analyzed_by_method'].keys():
    seen_boxes = 0
    accurately_detected_boxes = 0
    for seed in range(len(d['analyzed_by_method'][method_i])):#d['sucesses'][method_i]:
        boxes = d['analyzed_by_method'][method_i][seed]['boxes']
        accurately_detected_boxes += boxes[0]
        seen_boxes += boxes[1]
    method_boxes[method_i] = accurately_detected_boxes/seen_boxes

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)
ax.bar(np.array(list(method_boxes.keys())),
       method_boxes.values(),
       edgecolor = 'black',
       align='center', alpha=0.5,
       color = colors)
ax.set_xticks(np.array(list(d['N_failures'].keys())))
ax.set_ylabel('Correct Electrical Boxes Detection Accuracy at Terminal, %', fontdict={'fontsize': 18})
ax.set_xticklabels(['BPFS', 'BPFS-t', 'BPFS-tg', 'log-odds'])
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

#----------------------------------- Ground Truth MAPS difference -------------------------------
print(f"hamming average map dist {d['gt_maps_hamming']['norm_avg_dist']}")
# print(f"hamming min dist by map {d['gt_maps_hamming']['norm_min_dist_by_map']}")

print(f"jaccard average map dist {d['gt_maps_jaccard']['avg_dist']}")
# print(f"jaccard min dist by map {d['gt_maps_jaccard']['min_dist_by_map']}")

#--------------------------------------------------------------------------------------------------
from pingouin import ttest
import numpy as np
#https://ethanweed.github.io/pythonbook/05.02-ttest.html
results_12 = ttest(d['final_acc'][1], d['final_acc'][2], paired=True, alternative = 'greater')
results_13 = ttest(d['final_acc'][1], d['final_acc'][3], paired=True, alternative = 'greater')
results_14 = ttest(d['final_acc'][1], d['final_acc'][4], paired=True, alternative = 'greater')