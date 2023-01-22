import numpy as np
from numba import njit, prange
from .utils import T_from_pitch_yaw, distance_to_line, point_in_polygon
from .convex_hull import convex_hull_jarvis as convex_hull
from .import so1
import matplotlib.pyplot as plt

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

@njit(cache = True)
def minimal_distance_from_projected_boundry(ray_point : np.ndarray,
                                            projected_verts : np.ndarray):
    '''
    projected_verts - NX2 array [pitch,yaw] 
    ray_point - 1X2 [pitch,yaw]
    '''
    #add dimension
    hull_projected_verts = convex_hull(projected_verts)
    hull_projected_rots = np.zeros(hull_projected_verts.shape,dtype=np.complex128)

    #find rot_bar, mean approximation of the rotations
    #better optimization can be computed with https://github.com/dellaert/ShonanAveraging
    for i in prange(hull_projected_rots.shape[0]):
        hull_projected_rots[i] = so1.exp(hull_projected_verts[i])
    rot_query = so1.exp(ray_point)

    # rot_bar = so1.mu_rotations(hull_projected_rots)
    rot_bar = rot_query.copy()

    p = np.zeros_like(hull_projected_verts)
    for i in prange(hull_projected_rots.shape[0]):
        p[i] = so1.log(so1.minus(hull_projected_rots[i], rot_bar))
    q = so1.log(so1.minus(rot_query,rot_bar))

    p_plus = np.zeros((p.shape[0]+1,p.shape[1]))
    p_plus[:-1,:] = p
    p_plus[-1,:] = p[0]
    if point_in_polygon(q, p_plus):
        s, projected_point = distance_point_to_convex_hull(q, p_plus)
    else:
        s = 0.0
        projected_point = q

    # plt.figure()
    # plt.scatter(p_plus[:,0],p_plus[:,1])
    # plt.scatter(q[0],q[1])
    # plt.scatter(projected_point[0],projected_point[1])
    # plt.draw()

    dq = so1.log(so1.minus(so1.plus(rot_bar, so1.exp(projected_point)),rot_query))
    return s, dq