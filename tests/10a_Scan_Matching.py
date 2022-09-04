from numba import prange
import numpy as np
import matplotlib.pyplot as plt
import teaserpp_python
from bim4loc.sensors import Lidar1D
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(DIR_PATH, '10a_data')

sensor = Lidar1D(); 
sensor.std = 0.1;  
sensor.angles = np.linspace(-np.pi, np.pi, 36)
scan_to_points = sensor.get_scan_to_points()
sz_i = np.load(os.path.join(DATA_PATH,"scan_match - sz_i.npy"))
wz_i = np.load(os.path.join(DATA_PATH,"scan_match - wz_i.npy"))

n = sz_i.shape[0] #amount of lasers in scan
m = sz_i.shape[1] #amount of hits per laser
pwz_i = scan_to_points(wz_i)
psz_i = np.zeros((3, n * m))
pwz_i_bloated = np.zeros_like(psz_i)
for km in prange(m):
    psz_i[:, km * n : (km+1) * n] = scan_to_points(sz_i[:,km])
    pwz_i_bloated[:, km * n : (km+1) * n] = pwz_i

solver_params = teaserpp_python.RobustRegistrationSolver.Params()
solver_params.cbar2 = 10
solver_params.noise_bound = 0.3
solver_params.estimate_scaling = False
solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
solver_params.rotation_gnc_factor = 1.4
solver_params.rotation_max_iterations = 1000
solver_params.rotation_cost_threshold = 1e-12
print("Parameters are:", solver_params)

solver = teaserpp_python.RobustRegistrationSolver(solver_params)
# src = pwz_i_bloated
# dst = psz_i
src = pwz_i
dst = psz_i[:,:n]

solver.solve(src, dst)

solution = solver.getSolution()
print("Solution is:", solution)

src_T = solution.rotation @ src + solution.translation.reshape(-1,1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dst[0,:], dst[1,:], color = 'blue')
ax.scatter(src[0,:], src[1,:], color = 'red')
ax.scatter(src_T[0,:], src_T[1,:], color = 'purple', marker = 'x')
ax.axis('equal')
ax.legend(['dst', 'src', 'src_T'])
ax.set_title('red -> blue = purple')
plt.draw()
plt.show()