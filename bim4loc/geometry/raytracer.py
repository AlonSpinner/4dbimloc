import numpy as np
from numba import njit, prange
from typing import Union

EPS = 1e-16
NO_HIT = 2161354

@njit(parallel = True, cache = True)
def raytrace(rays : np.ndarray, meshes_v : np.ndarray, meshes_t : np.ndarray,
                    inc_v : int = 60, inc_t : int = 20,
                    max_hits : int = 10) -> Union[np.ndarray, np.ndarray]:
    '''
    input:
        rays - array of shape (n_rays, 6) containing [origin,direction]
        meshes_v - array of shape (n_meshes * inc_v, 3) containing vertices [px,py,pz]
        meshes_t - array of shape (n_meshes * inc_t, 3) containing triangles [id1,id2,id3]
        max_hits - after max_hits stop raytracing for ray.
                        this is an assumption that allows us to allocate size

        to clarify: 
            each mesh is contained of inc_v vertices and inc_t triangles
            if a triangle is [0,0,0] we will ignore it

    output:
        z_values - array of shape (n_rays, max_hits) containing range values
        z_ids - array of shape (n_rays, max_hits) containing meshes ids
                as provided in meshes_v and meshes_t
    '''
    N_meshes = int(meshes_t.shape[0]/ inc_t)
    N_rays = rays.shape[0]

    z_values = np.full((N_rays, max_hits), np.inf, dtype = np.float64)
    z_ids = np.full((N_rays, max_hits), NO_HIT, dtype = np.int32)

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

@njit(fastmath = True, cache = True)
def ray_triangle_intersection(ray : np.ndarray, triangle : np.ndarray) -> float:
    '''
    based on https://github.com/substack/ray-triangle-intersection/blob/master/index.js

    input:
        ray - np.array([x,y,z,vx,vy,vz])
        triangle - np.array([[ax,ay,az],
                            [bx,by,bz],
                            [cx,cy,cz]])

    output:
        z - distance to intersection             
    '''
    eye = ray[:3]
    dir = ray[3:]
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]

    pvec = np.cross(dir,edge2)
    det = np.dot(edge1,pvec)
    
    if det < EPS: #ray parallel to plane?
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


def post_process_raytrace(z_values : np.ndarray, z_ids : np.ndarray, 
                        solid_names : list[str], n_hits : int = 10) \
                            -> Union[np.ndarray, list[list[str]]]:
    '''
    input:
        z_values - array of shape (n_rays, max_hits) containing range values
        z_ids - array of shape (n_rays, max_hits) containing meshes ids
                as provided in meshes_v and meshes_t
        solid_names - list of strings to replace integers of z_ids
        n_hits - assumption of how many hits were provided per ray

    output:
        outputs are ordered: closest hit to furtherst hit
        pp_z_values - list of arrays. each row containing the range values of the ray hits
        pp_z_ids -  each element contains a list of the solid names that were hit by ray
    '''

    n_angles = z_values.shape[0]
    pp_z_values = np.full((n_angles, n_hits), np.inf, dtype = np.float64)
    pp_z_names = [['' for _ in range(n_hits)] for _ in range(n_angles)]

    for i_a, (zi_values, zi_ids) in enumerate(zip(z_values, z_ids)):
        ii_cond, = np.where(zi_values != np.inf)
        ii_sorted = np.argsort(zi_values[ii_cond])

        k = min(ii_sorted.size, n_hits)
        if k > 0:     
            pp_z_values[i_a] = zi_values[ii_sorted[:k]]
            for i_k in range(k):
                pp_z_names[i_a][i_k] = solid_names[zi_ids[ii_sorted[i_k]]]
    
    return pp_z_values, pp_z_names


if __name__ == "__main__":
    #simple test to show functionality and speed
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
