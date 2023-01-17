import numpy as np
from numba import njit, prange
from scipy.spatial import ConvexHull

def convex_hull(points : np.ndarray) -> np.ndarray:
    '''
    accepts and returns a numpy array of shape (N, 2)
    '''
    hull = ConvexHull(points)
    return points[hull.vertices]

@njit(cache = True)
def T_from_pitch_yaw(pitch, yaw):
    '''
    pitch - rotation around x axis
    yaw - rotation around y axis
    '''
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw), np.cos(yaw), 0],
                  [0, 0, 1]])
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    T = np.zeros((4,4)); T[3,3] = 1.0
    T[:3,:3] = R_yaw @ R_pitch
    return T

@njit(cache = True)
def project_to_ray(T_pose, ray_pitch, ray_yaw, mesh_v):
    T = T_pose @ T_from_pitch_yaw(ray_pitch, ray_yaw)
    projected_mesh_v = np.linalg.inv(T) @ mesh_v
    return projected_mesh_v

@njit(cache = True)
def minimal_distance_from_projected(projected_mesh_v : np.ndarray,
         point : np.ndarray = np.array([0,0])) -> float:
    '''
    projected_mesh_v - NX2 array 
    point - point to find the minimal distance to
    '''
    #go over lines and find the minimal distance from 0,0
    hull = convex_hull(projected_mesh_v)
    s = np.inf
    for i in prange(hull.shape[0]):
        s = min(s, distance_to_line(hull[i-1], hull[i], point))
    s = min(s, distance_to_line(hull[0], hull[-1], point)) #last line
    return s

@njit(cache = True)
def distance_to_line(p0,p1,q):
    #p0 and p1 are two points on the line
    #q is the point we want to find the distance to
    #https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    return np.linalg.norm(np.cross(p1-p0,q-p0))/np.linalg.norm(p1-p0)