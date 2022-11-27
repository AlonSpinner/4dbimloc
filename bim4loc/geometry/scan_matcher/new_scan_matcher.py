#inspirations:
# https://python.hotexamples.com/examples/cv2/-/estimateRigidTransform/python-estimaterigidtransform-function-examples.html
# https://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python

THRESHOLD_WEIGHT_FILTER = 0.3
import numpy as np
from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
from bim4loc.random.utils import negate
from bim4loc.geometry.raycaster import NO_HIT
from numba import njit, prange
import open3d as o3d
from bim4loc.geometry.pose2z import T_from_s

def scan_match(world_z, simulated_z, simulated_z_ids, 
                beliefs,
                sensor_std, sensor_max_range,
                sensor_scan_to_points,
                simulated_sensor_scan_to_points):

    #get points from scans
    qw = sensor_scan_to_points(world_z)
    qs = simulated_sensor_scan_to_points(simulated_z)

    #prep to filter, filtering max range points and low probability points
    fw_ids = world_z < sensor_max_range
    pzi_j = compute_weights(simulated_z_ids, beliefs)
    qs_weight = pzi_j.flatten()
    fs_ids = qs_weight > THRESHOLD_WEIGHT_FILTER

    src = qw[:, fw_ids] #from world
    dst = qs[:, fs_ids] #from simulated

    #robust ICP
    # R, t = robust_icp(src, dst, np.eye(4))
    errT = np.array([1.0, 0.0, 0.0, 0 *np.pi/8])
    R, t, used_src, used_dst = point2point_registration(src, dst, np.eye(4))
    plot(used_src, used_dst, R, t, False)
    return R,t

@njit(parallel = True, cache = True)
def compute_weights(simulated_z_ids : np.ndarray, beliefs : np.ndarray):
    '''
    simulated_z_ids: (n, m), n = number of rays, m = number of elements
    belifs (m) = number of elements
    '''
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

def robust_icp(src, dst, T0, threshold = 5, max_iter = 100):
    T_best = T0.copy()
    error_best = np.inf
    for i in range(max_iter):
        src_inds = np.random.choice(np.arange(src.shape[1]), 3, replace = False)
        dst_inds = np.random.choice(np.arange(dst.shape[1]), 3, replace = False)

        o3d_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src[:,src_inds].T))
        o3d_dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst[:,dst_inds].T))
        reg_p2p = o3d.pipelines.registration.registration_icp(
                        o3d_src, o3d_dst, threshold, T0,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint())

        if reg_p2p.inlier_rmse > 0.0 and reg_p2p.inlier_rmse < error_best:
            T_best = reg_p2p.transformation
            error_best = reg_p2p.inlier_rmse
    
    R = T_best[:3,:3]; t = T_best[:3,3].reshape(-1,1)
    return R, t

def point2point_registration(src, dst, T0, threshold = 10):
    o3d_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src.T))
    o3d_dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst.T))

    o3d_src = o3d_src.voxel_down_sample(voxel_size = 0.5)
    o3d_dst = o3d_dst.voxel_down_sample(voxel_size = 0.5)
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
                    o3d_src, o3d_dst, threshold, T0,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    T = reg_p2p.transformation
    R = T[:3,:3]; t = T[:3,3].reshape(-1,1)

    np_o3d_src = np.asarray(o3d_src.points)
    np_o3d_dst = np.asarray(o3d_dst.points)
    return R, t, np_o3d_src.T, np_o3d_dst.T

def plot(src, dst, R = np.eye(3), t = np.zeros((3,1)), 
            threeD = False, plot_correspondences = False):
    src_T = R @ src + t

    fig = plt.figure()

    if threeD:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(dst[0,:], dst[1,:], dst[2,:], color = 'blue', alpha = 0.5)
        ax.scatter(src[0,:], src[1,:], src[2,:], color = 'red', alpha = 0.5)
        ax.scatter(src_T[0,:], src_T[1,:], src_T[2,:], color = 'purple', marker = 'x')
        ax.axis('auto')
    else:
        ax = fig.add_subplot(111)
        ax.scatter(dst[0,:], dst[1,:], color = 'blue', alpha = 0.5)
        ax.scatter(src[0,:], src[1,:], color = 'red', alpha = 0.5)
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