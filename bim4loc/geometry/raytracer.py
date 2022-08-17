import numpy as np
from numba import njit, prange
import numba
from numba.typed import Dict
from numba.core import types
from dataclasses import dataclass
from bim4loc.maps import RayTracingMap
from collections import namedtuple

EPS = 1e-16
NO_HIT = 2161354
SceneType = namedtuple('scene', ['vertices', 
                                'triangles', 
                                'inc_v', 
                                'inc_t'])

def map2scene(m : RayTracingMap):   
    max_vertices = 0
    max_triangles = 0
    for s in m.solids.values():
        n_v = (np.asarray(s.geometry.vertices)).shape[0]
        n_t = (np.asarray(s.geometry.triangles)).shape[0]

        if n_v > max_vertices:
            max_vertices = n_v
        if n_t > max_triangles:
            max_triangles = n_t

    inc_v = max_vertices
    inc_t = max_triangles

    n = len(m.solids)
    meshes_v = np.zeros((n * inc_v,3), dtype = np.float64)
    meshes_t = np.zeros((n * inc_t,3), dtype = np.int32)
    for i_s, s in enumerate(m.solids.values()):
        v = np.asarray(s.geometry.vertices)
        t = np.asarray(s.geometry.triangles)
        meshes_v[i_s * inc_v : i_s * inc_v + v.shape[0]] = v
        meshes_t[i_s * inc_t : i_s * inc_t + t.shape[0]] = t

    return SceneType(meshes_v, meshes_t, inc_v, inc_t)

def post_process_raytrace(z_values, z_ids, solid_names, n_hits = 0):
    pp_z_values = []
    pp_z_names = []
    for zi_values, zi_ids in zip(z_values, z_ids):
        ii_cond, = np.where(zi_values != np.inf)
        ii_sorted = np.argsort(zi_values[ii_cond])
        
        if n_hits > 0: #assumes all rays have at least n_hits
            ii_sorted = ii_sorted[:n_hits]

        pp_z_values.append(zi_values[ii_sorted])
        pp_z_names.append([solid_names[i] for i in zi_ids[ii_sorted]])
    
    return pp_z_values, pp_z_names

@njit(parallel = True)
def raytrace(rays : np.ndarray, meshes_v : np.ndarray, meshes_t : np.ndarray,
                    inc_v : int = 60, inc_t : int = 20,
                    max_hits : int = 10) -> None:
    N_meshes = int(meshes_t.shape[0]/ inc_t)
    N_rays = rays.shape[0]

    z_values = np.full((N_rays, max_hits), np.inf, dtype = np.float64)
    z_ids = np.full((N_rays, max_hits), NO_HIT, dtype = np.int32)

    #assumes meshes have no more than 20 triangles and no more than 60 vertices
    for i_r in prange(N_rays):
        ray_max_hits = False
        ray = rays[i_r]
        i_hit = 0

        for i_m in prange(N_meshes):
            finished_mesh = False

            m_t = meshes_t[i_m * inc_t : (i_m + 1) * inc_t]
            m_v = meshes_v[i_m * inc_v : (i_m + 1) * inc_v]
            for i_t in prange(m_t.shape[0]):
                triangle = m_v[m_t[i_t]]
                if triangle.sum() == 0: #empty triangle
                    finished_mesh = True
                    break
                z = ray_triangle_intersection(ray, triangle)
                if z != NO_HIT and z > 0:
                    z_values[i_r, i_hit] = z
                    z_ids[i_r, i_hit] = i_m
                    i_hit += 1

                if i_hit == max_hits:
                    ray_max_hits = True
                    break
            
            if ray_max_hits or finished_mesh:
                break
    
    return z_values, z_ids

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
        return NO_HIT
    
    tvec  = eye - triangle[0]
    u = np.dot(tvec, pvec)
    if u < 0 or u > det:
        return NO_HIT
    qvec = np.cross(tvec,edge1)
    v = np.dot(dir,qvec)
    if v < 0 or u + v > det:
        return NO_HIT

    z = np.dot(edge2,qvec) / det
    return z

if __name__ == "__main__":
    import time
    ray = np.array([0,0,0,1,0,0],dtype=float)
    triangle = np.array([[2,-1,-1],
                        [2,0,1],
                        [2,1,-1]], dtype = float)
    
    N = int(1e4)
    ray_triangle_intersection(ray, triangle)
    s = time.time()
    for _ in range(N):
        ray_triangle_intersection(ray, triangle)
    e = time.time()
    print((e-s)/N)
    # ray_triangle_intersection.parallel_diagnostics(level = 4)