from numba import prange
import numpy as np
import matplotlib.pyplot as plt
import teaserpp_python

def scan_match(wz_i, sz_i, szid_i, beliefs, scan_to_points, sensor_std, sensor_max_range):
    #convert scan to points
    pwz_i = scan_to_points(wz_i)

    n = szid_i.shape[0] #amount of lasers in scan
    m = szid_i.shape[1] #amount of hits per laser
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
    src = pwz_i_bloated
    dst = psz_i
    # src = pwz_i
    # dst = psz_i[:,:n]
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

def icp(src, dst, weights) -> np.ndarray:
    #src and dst are 3xN
    #weights are 1xN
    #return 4x4 matrix
    #src = src + np.random.normal(0, 0.1, src.shape)
    #dst = dst + np.random.normal(0, 0.1, dst.shape)
    #weights = np.ones_like(weights)
    weights = weights / np.sum(weights)
    src = src - np.sum(src * weights, axis = 1, keepdims = True)
    dst = dst - np.sum(dst * weights, axis = 1, keepdims = True)
    cov = src @ dst.T * weights
    u, s, vh = np.linalg.svd(cov)
    R = vh.T @ u.T
    t = np.sum(dst * weights, axis = 1, keepdims = True) - R @ np.sum(src * weights, axis = 1, keepdims = True)
    return np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))

