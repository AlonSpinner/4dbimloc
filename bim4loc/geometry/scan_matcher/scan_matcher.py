import numpy as np
import matplotlib.pyplot as plt
from bim4loc.random.utils import negate
from bim4loc.geometry.raycaster import NO_HIT
import open3d as o3d
from numba import njit, prange
from scipy.spatial import cKDTree

#import teaser only if installed
import importlib
teaser_loader = importlib.util.find_spec("teaserpp_python")
if teaser_loader is not None:
    import teaserpp_python

THRESHOLD_WEIGHT_FILTER = 0.4

def scan_match(world_z, simulated_z, simulated_z_ids, simulated_z_normals, 
                beliefs,
                sensor_std, sensor_max_range,
                scan_to_points):
 
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
                                # np.eye(4), threshold = 0.5, k = sensor_std)
    # R, t = point2point_registration(src, dst, np.eye(4), threshold = 0.5)
    # R, t = teaser_registration(src, dst)
    R, t = point2point_ransac(src, dst, normals)

    plot(src, dst, R, t, False)
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

def estimate_normals(o3d_pcd, voxel_size = 0.2):
    o3d_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    return o3d_pcd

def compute_fpfh(o3d_pcd, voxel_size = 0.2):
    #http://www.open3d.org/docs/release/python_example/pipelines/index.html#icp-registration-py
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        o3d_pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                            max_nn=100))
    return pcd_fpfh

def point2point_ransac(src, dst, normals, 
                        threshold = 0.5, 
                        RANASC_max_iterations = 1000,
                        RANSAC_confidence = 0.99):

    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src.T))
    src = src.voxel_down_sample(0.2)
    src = estimate_normals(src)
    src_fpfh = compute_fpfh(src)
    
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst.T))
    # dst.normals = o3d.utility.Vector3dVector(normals)
    dst = dst.voxel_down_sample(0.2)
    dst = estimate_normals(dst)
    dst_fpfh = compute_fpfh(dst)
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src,
        dst,
        src_fpfh,
        dst_fpfh,
        mutual_filter = True,
        max_correspondence_distance = threshold,
        estimation_method=o3d.pipelines.registration.
        TransformationEstimationPointToPoint(False), #without scaling
        ransac_n= 3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            RANASC_max_iterations, RANSAC_confidence))
    T =  result.transformation
    R = T[:3,:3]; t = T[:3,3].reshape(-1,1)
    return R, t

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=knn, n_jobs=-1)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
  nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
  corres01_idx0 = np.arange(len(nns01))
  corres01_idx1 = nns01

  if not mutual_filter:
    return corres01_idx0, corres01_idx1

  nns10 = find_knn_cpu(feats1, feats0, knn=1)
  corres10_idx1 = np.arange(len(nns10))
  corres10_idx0 = nns10

  mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
  corres_idx0 = corres01_idx0[mutual_filter]
  corres_idx1 = corres01_idx1[mutual_filter]

  return corres_idx0, corres_idx1

def teaser_registration(src, dst):
    src_np = src
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src.T))
    src = src.voxel_down_sample(0.2)
    src = estimate_normals(src)
    src_fpfh = compute_fpfh(src); src_fpfh = np.array(src_fpfh.data).T

    dst_np = dst
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst.T))
    dst = dst.voxel_down_sample(0.2)
    dst = estimate_normals(dst)
    dst_fpfh = compute_fpfh(dst); dst_fpfh = np.array(dst_fpfh.data).T

    src_corrs, dst_corrs = find_correspondences(
    src_fpfh, dst_fpfh, mutual_filter=True)

    src = src_np[:, src_corrs]
    dst = dst_np[:, dst_corrs]
    plot(src, dst, plot_correspondences = True)

    teaser_solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    teaser_solver_params.cbar2 = 1
    teaser_solver_params.noise_bound = 0.01
    teaser_solver_params.estimate_scaling = False
    teaser_solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    teaser_solver_params.rotation_gnc_factor = 1.4
    teaser_solver_params.rotation_max_iterations = 10000
    teaser_solver_params.rotation_cost_threshold = 1e-16
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

def plot(src, dst, R = np.eye(3), t = np.zeros((3,1)), 
            threeD = False, plot_correspondences = False):
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
        if plot_correspondences:
            for s,d in zip(src.T,dst.T):
                ax.plot([s[0],d[0]], [s[1],d[1]], color = 'black')
        ax.axis('equal')
    ax.grid('on')
    ax.legend(['dst', 'src', 'src_T'])
    ax.set_title('red -> blue = purple')
    plt.draw()
    plt.show()



