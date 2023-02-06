import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from bim4loc.random.one_dim import Gaussian

plt.rcParams['font.size'] = '24'
def plot_gaussian(ax : plt.Axes, gauss : Gaussian, color = 'b', alpha = 0.7, lims = None):
    if lims == None:
        tmin = gauss.mu - 3 * gauss.sigma
        tmax = gauss.mu + 3 * gauss.sigma
        t = np.linspace(tmin,tmax,1000)
    else:
        t = np.linspace(lims[0],lims[1],1000)
    y = gauss.pdf(t)
    ax.plot(t,y, color = color, alpha = alpha)

dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path ,"analyzed_data.p")
d = pickle.Unpickler(open(file, "rb")).load()

colors = ['b', 'g', 'r', 'k']
#-----------------------------------Failures --------------------------------------------
fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)
ax_f = ax.bar(np.array(list(d['N_traj_failures'].keys())),
       d['N_failures'].values(),
       edgecolor = 'black',
       align='center', alpha=1.0,
       color = colors)
ax.set_xticks(np.array(list(d['N_failures'].keys())))
ax.set_ylabel('Failures')
ax.set_xticklabels(['BPFS', 'BPFS-t', 'BPFS-tg', 'log-odds'])
plt.show()

#----------------------------------- Mean Trajectory Error ------------------------------------
fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)

for i, m in enumerate(d['mean_traj_err'].values()):
    g = Gaussian(np.mean(m),np.std(m))
    plot_gaussian(ax, g, color = colors[i], alpha = 1.0, lims = [0.0, 0.4])
ax.set_xlabel('Mean Trajectory Error, m')
ax.set_ylabel('Probability Density')
ax.grid(True)
plt.show()
#----------------------------------- Mean Cross Entropy Error ------------------------------------
fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)
for i, m in enumerate(d['final_mean_ce_err'].values()):
    g = Gaussian(np.mean(m),np.mean(m))
    plot_gaussian(ax, g, color = colors[i], alpha = 1.0, lims = [0.0,9.0])
ax.set_xlabel('Mean Final Cross Entropy Error')
ax.set_ylabel('Probability Density')
ax.grid(True)
plt.show()
#----------------------------------- Electric Boxes ------------------------------------
boxes_right = {}
for method_i in d['analyzed_by_method'].keys():
    p_seeds = []
    for seed in d['sucesses'][method_i]:
        p_seeds.append(d['analyzed_by_method'][method_i][seed]['percentile_boxes'])
    boxes_right[method_i] = np.mean(p_seeds)

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)
ax.bar(np.array(list(boxes_right.keys())),
       boxes_right.values(),
       edgecolor = 'black',
       align='center', alpha=1.0,
       color = colors)
ax.set_xticks(np.array(list(boxes_right.keys())))
ax.set_xticklabels(['BPFS', 'BPFS-t', 'BPFS-tg', 'log-odds'])
ax.set_ylabel('Correct Wall-Hung Box Detection, %')
plt.show()

#----------------------------------- Accuracy ------------------------------------
fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)
ax_f = ax.bar(np.array(list(d['final_acc'].keys())),
       [np.mean(v) for v in d['final_acc'].values()],
       edgecolor = 'black',
       align='center', alpha=1.0,
       color = colors)
ax.set_xticks(np.array(list(d['final_acc'].keys())))
ax.set_ylabel('Accuracy, %')
ax.set_xticklabels(['BPFS', 'BPFS-t', 'BPFS-tg', 'log-odds'])
plt.show()