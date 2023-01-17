import numpy as np
from numba import njit, prange
from .utils import T_from_pitch_yaw, distance_to_line, pca
from .convex_hull import convex_hull_chan as convex_hull
from .import so3

@njit(cache = True)
def project_mesh_to_ray(T_pose, ray_pitch, ray_yaw, mesh_v):
    T = T_pose @ T_from_pitch_yaw(ray_pitch, ray_yaw)
    projected_mesh_v = np.linalg.inv(T) @ mesh_v
    return projected_mesh_v


@njit(cache = True)
def distance_point_to_convex_hull(point, hull_plus):
    #linear search
    s = np.inf
    projected_point = np.zeros(2)
    for i in range(1,hull_plus.shape[0]):
        test_s, test_projected_point =  distance_to_line(hull_plus[i-1], hull_plus[i], point)
        if test_s < s:
            s = test_s
            projected_point = test_projected_point
    return s, projected_point

# @njit(cache = True)
def minimal_distance_from_projected_boundry(ray_point : np.ndarray,
                                            projected_verts : np.ndarray):
    '''
    projected_verts - NX2 array [pitch,yaw] 
    ray_point - 1X2 [pitch,yaw]
    '''
    #add dimension
    projected_verts = np.hstack((np.zeros((projected_verts.shape[0],1)), projected_verts)) #roll-pitch-yaw
    projected_rots = np.zeros((projected_verts.shape[0],3,3))

    #find rot_bar, mean approximation of the rotations
    #better optimization can be computed with https://github.com/dellaert/ShonanAveraging
    for i in prange(projected_verts.shape[0]):
        projected_rots[i] = so3.exp(projected_verts[i])
    rot_bar = so3.mu_rotations(projected_rots)

    query = np.zeros(3)
    query[1:] = ray_point
    rot_query = so3.exp(query)

    p = np.array([so3.log(so3.minus(rot_bar,rot)) for rot in projected_rots])
    q = so3.log(so3.minus(rot_bar,rot_query))

    #because the convex hull is in 2D, we need to project the vertices onto the plane
    phat, transform = pca(p)
    qhat = q @ transform

    hull = convex_hull(np.array(phat))
    hull_plus = np.vstack((hull,hull[0]))
    s, projected_point = distance_point_to_convex_hull(qhat, hull_plus)

    projected_point = projected_point @ transform.T
    projected_point = so3.log(so3.plus(rot_bar, so3.exp(projected_point)))
    return s, projected_point