from numba import prange
import numpy as np
import matplotlib.pyplot as plt
import teaserpp_python
from bim4loc.random.utils import negate
from bim4loc.geometry.raycaster import NO_HIT
from bim4loc.geometry.poses import Pose2z
import open3d as o3d

def scan_match(wz_i, sz_i, szid_i, beliefs, pose : Pose2z, pose_real : Pose2z, scan_to_points):

    #weight of each point pair ~ probability of hitting the solid / max range
    pzi_j = compute_weights(szid_i, beliefs)
    
    #convert scan to points
    qwz_i = scan_to_points(wz_i)

    n = szid_i.shape[0] #amount of lasers in scan
    m = szid_i.shape[1] #amount of hits per laser
    qsz_i = np.zeros((3, n * m))
    qwz_i_bloated = np.zeros_like(qsz_i)
    weights = np.zeros(n * m)

    for km in prange(m): #this is wierd, but we loop over amount of hits
        qsz_i[:, km * n : (km+1) * n] = scan_to_points(sz_i[:,km])
        qwz_i_bloated[:, km * n : (km+1) * n] = qwz_i #<---- why we loop over amount of hits
        weights[km * n : (km+1) * n] = pzi_j[:,km]
                
    # src = qwz_i_bloated
    # dst = pose.transform_from(qsz_i) #sensor system to world system
    
    # src = pose_real.transform_from(qwz_i)
    # dst = pose.transform_from(qsz_i[:,:n])
    src = qwz_i
    dst = qsz_i[:,:n]
    weights = weights[:n]
    
    bools = np.linalg.norm(src - dst, axis = 0) < 4.0
    src = src[:, bools]
    dst = dst[:, bools]
    weights = weights[bools]

    # R, t = weighted_registration(src, dst, np.ones_like(weights))
    # src_T = R @ src + t

    o3d_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src.T))
    # src.estiamte_normals()
    o3d_dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst.T))
    # dst.estiamte_normals()
    # loss = o3d.pipelines.registration.TukeyLoss(k = 0.3)
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    trans_init = np.eye(4)
    threshold = 2
    reg_p2l = o3d.pipelines.registration.registration_icp(
        o3d_src, o3d_dst, threshold, trans_init, p2l)
    T = reg_p2l.transformation
    R = T[:3,:3]; t = T[:3,3].reshape(-1,1)
    src_T = R @ src + t

    # solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    # solver_params.cbar2 = 4
    # solver_params.noise_bound = 0.01
    # solver_params.estimate_scaling = False
    # solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    # solver_params.rotation_gnc_factor = 1.4
    # solver_params.rotation_max_iterations = 1000
    # solver_params.rotation_cost_threshold = 1e-12
    # print("Parameters are:", solver_params)

    # solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    # solver.solve(src, dst)
    # solution = solver.getSolution()
    # print("Solution is:", solution)
    # src_T = solution.rotation @ src + solution.translation.reshape(-1,1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dst[0,:], dst[1,:], color = 'blue')
    ax.scatter(src[0,:], src[1,:], color = 'red')
    ax.scatter(src_T[0,:], src_T[1,:], color = 'purple', marker = 'x')
    for s, d in zip(src.T, dst.T):
        ax.plot([s[0], d[0]], [s[1], d[1]], color = 'black')
    ax.axis('equal')
    ax.legend(['dst', 'src', 'src_T'])
    ax.set_title('red -> blue = purple')
    plt.draw()
    plt.show()

def compute_weights(szid, beliefs):
    N_maxhits = szid.shape[1]
    N_rays = szid.shape[0]
    
    pzi = np.zeros_like(szid, dtype = np.float32)
    for i in prange(N_rays):
        Pbar = 1.0
        for j in prange(N_maxhits):
            if szid[i,j] == NO_HIT:
                pzi[i,j] = Pbar #probablity for max hit
                break
            
            pzi[i,j] = Pbar * beliefs[szid[i,j]]
            Pbar = Pbar * negate(beliefs[szid[i,j]])

    return pzi

def weighted_registration(src, dst, weights) -> np.ndarray:
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



