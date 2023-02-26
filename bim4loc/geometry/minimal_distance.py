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

    q = so1.exp(ray_point)
    p = np.zeros((projected_verts.shape[0],2), dtype = np.complex128)
    for i in prange(projected_verts.shape[0]):
        p[i] = so1.exp(projected_verts[i])

    pmq = np.zeros_like(p)
    for i in prange(p.shape[0]):
        pmq[i] = so1.minus(p[i],q)

    # plt.figure()
    # t = np.linspace(0,2*np.pi,100)
    # plt.plot(np.cos(t),np.sin(t))
    # plt.scatter(np.real(q[1]),np.imag(q[1]), color = 'r')
    # plt.scatter(np.real(p[:,1]),np.imag(p[:,1]))
    # plt.draw()
    
    dpmq = convex_hull(so1.log(pmq))

    dpmq_plus = np.zeros((dpmq.shape[0]+1,dpmq.shape[1]))
    dpmq_plus[:-1,:] = dpmq
    dpmq_plus[-1,:] = dpmq[0]
    dpmq_plus[:,1] *= 2 #<------------------- make pitch distance twice as significant
    if point_in_polygon(np.zeros(2), dpmq):
        s, projected_point = distance_point_to_convex_hull(np.zeros(2), dpmq_plus)
    else:
        s = 0.0
        projected_point = np.zeros(2)

    # plt.figure()
    # plt.scatter(projected_point[0],projected_point[1], color = 'r')
    # plt.plot(dpmq[:,0],dpmq[:,1])
    # plt.draw()

    return s, projected_point