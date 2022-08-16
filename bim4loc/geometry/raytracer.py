import numpy as np
from numba import njit, prange
import numba
from numba.typed import Dict
from numba.core import types
from dataclasses import dataclass

EPS = 1e-16

# @dataclass(frozen = True)
# class Mesh:
#     pass

# @njit(cache = True, parallel = True)
# def raytrace(rays : np.ndarray, meshes : list[Mesh]) -> None:
#     #assumes meshes have no more than 20 triangles and no more than 60 vertices
#     for i_r in prange(rays.shape[0]):
#         ray = rays[i_r]
#         for mesh, m in prange(meshes.shape[0]):
#             for j in prange(mesh.triangles.shape[0]):
#                 triangle = mesh.triangles[j]
#                 z = ray_triangle_intersection(ray, triangle)
#                 if z:
#                     rays[i, 2] = [z,
                    

                    
#             else:
#                 continue
#             break
#         else:
#             continue
#         break


#     return
@numba.njit(fastmath = True)
def ray_triangle_intersection(ray : np.ndarray, triangle : np.ndarray) -> float:
    '''
    based on https://github.com/substack/ray-triangle-intersection/blob/master/index.js

    ray - np.array([x,y,z,vx,vy,vz])
    triangle - np.array([[ax,ay,az],
                        [bx,by,bz],
                        [cx,cy,cz]])
                        
    '''
    eye = ray[:3]
    dir = ray[3:]
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]

    pvec = np.cross(dir,edge2)
    det = np.dot(edge1,pvec)
    
    if det < EPS: #ray parallel to normal?
        return 0
    
    tvec  = eye - triangle[0]
    u = np.dot(tvec, pvec)
    if u < 0 or u > det:
        return 0
    qvec = np.cross(tvec,edge1)
    v = np.dot(dir,qvec)
    if v < 0 or u + v > det:
        return 0

    z = np.dot(edge2,qvec) / det
    return z

ray = np.array([0,0,0,1,0,0],dtype=float)
triangle = np.array([[2,-1,-1],
                    [2,0,1],
                    [2,1,-1]], dtype = float)


import time
N = int(1e4)
s = time.time()
for _ in range(N):
    ray_triangle_intersection(ray, triangle)
e = time.time()

print((e-s)/N)
# ray_triangle_intersection.parallel_diagnostics(level = 4)