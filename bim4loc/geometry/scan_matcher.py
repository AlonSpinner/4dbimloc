import numpy as np
import matplotlib.pyplot as plt
from bim4loc.random.utils import negate
from bim4loc.geometry.raycaster import NO_HIT
import open3d as o3d
from numba import njit, prange

#import teaser only if installed
import importlib
teaser_loader = importlib.util.find_spec("teaserpp_python")
if teaser_loader is not None:
    import teaserpp_python

THRESHOLD_WEIGHT_FILTER = 0.4

def scan_match(world_z, simulated_z, simulated_z_ids, simulated_z_normals, 
                beliefs,
                sensor_std, sensor_max_range,
                scan_to_points, errT):
 
    #convert scan to points
    qw = scan_to_points(world_z)
    qs = scan_to_points(simulated_z)

    #prep to filter
    fw_ids = world_z < sensor_max_range
    pzi_j = compute_weights(simulated_z_ids, beliefs)
    qs_weight = pzi_j.flatten()
    fs_ids = qs_weight > THRESHOLD_WEIGHT_FILTER

    src = qw[:, fw_ids]; 
    dst = qs[:, fs_ids]; 
    normals = simulated_z_normals.reshape(-1, 3)[fs_ids]

    #kill third dimension?
    # src[2,:] = 0.0
    # dst[2,:] = 0.0

    # R, t = point2plane_registration(src, dst, normals, 
    #                             np.eye(4), threshold = 0.5, k = sensor_std)
    R, t = point2point_registration(src, dst, errT, threshold = 3)

    plot(src, dst, R, t, True)
    return R,t

@njit(parallel = True, cache = True)
def compute_weights(simulated_z_ids, beliefs):
    #note: WE DO NOT USE WORLD_Z HERE AS WE CANNOT CORRELATE WORLD_Z TO SIMULATED_Z
    N_maxhits = simulated_z_ids.shape[1]
    N_rays = simulated_z_ids.shape[0]
    
    pzij = np.zeros_like(simulated_z_ids, dtype = np.float32)
    for i in prange(N_rays):
        Pbar = 1.0
        for j in prange(N_maxhits):
            if simulated_z_ids[i,j] == NO_HIT:
                #don't use max range measurements!
                break
            
            pzij[i,j] = Pbar * beliefs[simulated_z_ids[i,j]]
            Pbar = Pbar * negate(beliefs[simulated_z_ids[i,j]])

    return pzij

def teaser_registration(src, dst):
    teaser_solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    teaser_solver_params.cbar2 = 4
    teaser_solver_params.noise_bound = 0.01
    teaser_solver_params.estimate_scaling = False
    teaser_solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    teaser_solver_params.rotation_gnc_factor = 1.4
    teaser_solver_params.rotation_max_iterations = 1000
    teaser_solver_params.rotation_cost_threshold = 1e-12
    solver = teaserpp_python.RobustRegistrationSolver(teaser_solver_params)

    solver.solve(src, dst)
    solution = solver.getSolution()
    return solution.rotation, solution.translation.reshape(-1,1)

def point2point_registration(src, dst, T0, threshold = 0.5):
    o3d_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src.T))
    o3d_dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst.T))
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
                    o3d_src, o3d_dst, threshold, T0,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    T = reg_p2p.transformation
    R = T[:3,:3]; t = T[:3,3].reshape(-1,1)
    return R, t

def point2plane_registration(src, dst, normals, 
                                T0, threshold = 0.5, k = 0.1):
    o3d_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src.T))
    o3d_dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst.T))
    o3d_dst.normals = o3d.utility.Vector3dVector(normals)
    
    loss = o3d.pipelines.registration.TukeyLoss(k = k)
    reg_p2l = o3d.pipelines.registration.registration_icp(
                    o3d_src, o3d_dst, threshold, T0,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(loss))
    
    T = reg_p2l.transformation
    R = T[:3,:3]; t = T[:3,3].reshape(-1,1)
    return R, t

def weighted_registration(src, dst, weights):
    '''
    based on "Least Squares Rigid Motion Using SVD" 
    by Olga Sorkine-Hornung and Michael Rabinovich

    src and dst are 3xN
    weights are 1xN    
    returns transformation such that T(src) ~ dst
    '''
    s_weights = np.sum(weights)

    src_bar = np.sum((weights * src), axis = 1, keepdims = True) / s_weights
    dst_bar = np.sum((weights * dst), axis = 1, keepdims = True) / s_weights

    X = src - src_bar
    Y = dst - dst_bar
    W = np.diag(weights)

    S = X @ W @ Y.T

    U, Sigma, Vh = np.linalg.svd(S)
    V = Vh.T

    R = V @ np.diag([1.0, 1.0, np.linalg.det(V @ U.T)]) @ U.T
    t = dst_bar - R @ src_bar
    return R, t

def plot(src, dst, R, t, threeD = False):
    src_T = R @ src + t

    fig = plt.figure()

    if threeD:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(dst[0,:], dst[1,:], dst[2,:], color = 'blue')
        ax.scatter(src[0,:], src[1,:], src[2,:], color = 'red')
        ax.scatter(src_T[0,:], src_T[1,:], src_T[2,:], color = 'purple', marker = 'x')
        ax.axis('auto')
    else:
        ax = fig.add_subplot(111)
        ax.scatter(dst[0,:], dst[1,:], color = 'blue')
        ax.scatter(src[0,:], src[1,:], color = 'red')
        ax.scatter(src_T[0,:], src_T[1,:], color = 'purple', marker = 'x')
        ax.axis('equal')
    ax.legend(['dst', 'src', 'src_T'])
    ax.set_title('red -> blue = purple')
    plt.draw()
    plt.show()



