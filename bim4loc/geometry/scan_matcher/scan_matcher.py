#inspirations:
# https://python.hotexamples.com/examples/cv2/-/estimateRigidTransform/python-estimaterigidtransform-function-examples.html
# https://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python

import numpy as np
from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
from bim4loc.random.utils import negate
from bim4loc.geometry.raycaster import NO_HIT
from numba import njit, prange
import open3d as o3d

def dead_reck_scan_match(T0, z_prev, z, sensor_max_range,
                sensor_scan_to_points,
                downsample_voxelsize = 0.5,
                icp_distance_threshold = 10.0):
    #get points from scans
    q = sensor_scan_to_points(z_prev)
    p = sensor_scan_to_points(z)

    #prep to filter, filtering max range points and low probability points
    fq_ids = z_prev < sensor_max_range
    fp_ids = z < sensor_max_range

    src = q[:, fq_ids] #from prev
    dst = p[:, fp_ids] #from current

    #point to point
    src, dst = preprocess_points(src, dst, 
                            voxelsize = downsample_voxelsize)
    R, t, rmse = point2point_registration(src, dst, T0, 
                            distance_threshold = icp_distance_threshold)
    
    #for debugging purposes, when running 10
    # plot(np.asarray(src.points).T, np.asarray(dst.points).T, R, t, rmse, False)
    
    #adding [R,t] to the simulated pose to get better estimate for world pose
    return R, t, rmse

def scan_match(world_z, simulated_z, simulated_z_ids,
                beliefs,
                sensor_std, sensor_max_range,
                sensor_scan_to_points,
                simulated_sensor_scan_to_points,
                downsample_voxelsize = 0.5,
                icp_distance_threshold = 10.0,
                probability_filter_threshold = 0.3):

    #get points from scans
    qw = sensor_scan_to_points(world_z)
    qs = simulated_sensor_scan_to_points(simulated_z)

    #prep to filter, filtering max range points and low probability points
    fw_ids = world_z < sensor_max_range
    pzi_j = compute_weights(simulated_z_ids, beliefs)
    qs_weight = pzi_j.flatten()
    fs_ids = qs_weight > probability_filter_threshold

    src = qw[:, fw_ids] #from world
    dst = qs[:, fs_ids] #from simulated

    #point to point
    src, dst = preprocess_points(src, dst, 
                            voxelsize = downsample_voxelsize)
    R, t, rmse = point2point_registration(src, dst, np.eye(4), 
                            distance_threshold = icp_distance_threshold)
    
    #for debugging purposes, when running 10
    # plot(np.asarray(src.points).T, np.asarray(dst.points).T, R, t, rmse, False)
    
    #adding [R,t] to the simulated pose to get better estimate for world pose
    return R, t, rmse

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

def preprocess_points(src : np.ndarray, dst : np.ndarray,
                        dst_normals = None, voxelsize = 0.5):
    o3d_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src.T))
    o3d_dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst.T))
    if dst_normals is not None:
        o3d_dst.normals = o3d.utility.Vector3dVector(dst_normals)

    o3d_src = o3d_src.voxel_down_sample(voxel_size = voxelsize)
    o3d_dst = o3d_dst.voxel_down_sample(voxel_size = voxelsize)
    return o3d_src, o3d_dst

#-------------------------------------------------------------------------
#---------------------------REGISTRATION METHODS--------------------------
#-------------------------------------------------------------------------

def point2point_registration(o3d_src, o3d_dst, T0,  distance_threshold = 10.0):
    #can also use
    # open3d.pipelines.registration.registration_icp
    reg = o3d.pipelines.registration.registration_icp(
                    o3d_src, o3d_dst, distance_threshold, T0,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10))
    
    T = reg.transformation
    R = T[:3,:3]; t = T[:3,3].reshape(-1,1)
    return R, t, reg.inlier_rmse

def point2plane_registration(o3d_src, o3d_dst, 
                                T0,  distance_threshold = 10.0, k = 5):
    
    #did not work out so well
    loss = o3d.pipelines.registration.TukeyLoss(k = k)
    reg = o3d.pipelines.registration.registration_icp(
                    o3d_src, o3d_dst, distance_threshold, T0,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10))

    T = reg.transformation
    R = T[:3,:3]; t = T[:3,3].reshape(-1,1)
    return R, t, reg.inlier_rmse

def gicp_registration(o3d_src, o3d_dst, T0, distance_threshold = 10.0):
    reg = o3d.pipelines.registration.registration_generalized_icp(
                    o3d_src, o3d_dst, distance_threshold, T0)
    
    T = reg.transformation
    R = T[:3,:3]; t = T[:3,3].reshape(-1,1)
    return R, t, reg.inlier_rmse

#-------------------------------------------------------------------------
#----------------------------DEBUGGING TOOL-------------------------------
#-------------------------------------------------------------------------


def plot(src, dst, R = np.eye(3), t = np.zeros((3,1)), rmse = 0.0, 
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
    ax.set_title(f'red -> blue = purple\nrmse = {rmse}')
    plt.draw()
    plt.show()