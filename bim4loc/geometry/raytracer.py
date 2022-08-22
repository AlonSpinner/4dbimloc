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
        outputs are sorted by z_values (small to big)
    '''
    N_meshes = int(meshes_t.shape[0]/ inc_t)
    N_rays = rays.shape[0]

    z_values = np.full((N_rays, max_hits), np.inf, dtype = np.float64)
    z_ids = np.full((N_rays, max_hits), NO_HIT, dtype = np.int32)

    for i_r in prange(N_rays):
        ray = rays[i_r]

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
                    ii = np.searchsorted(z_values[i_r], z)
                    if ii == max_hits: #new z is the biggest, throw it away
                        pass
                    elif ii == 0:
                        z_values[i_r] = np.hstack((np.array([z]), z_values[i_r][1:]))
                        z_ids[i_r] = np.hstack((np.array([i_m]), z_ids[i_r][1:]))
                    else:
                        z_values[i_r] = np.hstack((z_values[i_r][:ii], np.array([z]), z_values[i_r][ii:-1]))
                        z_ids[i_r] = np.hstack((z_ids[i_r][:ii], np.array([i_m]), z_ids[i_r][ii:-1]))
            
            if finished_mesh:
                break
    
    for i_r in prange(N_rays):
        i_sorted = np.argsort(z_values[i_r])
        z_values[i_r] = z_values[i_r][i_sorted]
        z_ids[i_r] = z_ids[i_r][i_sorted]
    
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
    if det < EPS: #abs(det) < EPS if we want internal intersection
        return NO_HIT
    
    inv_det = 1.0/det
    tvec  = eye - triangle[0]
    u = np.dot(tvec, pvec) * inv_det
    if u < 0 or u > 1.0:
        return NO_HIT
    qvec = np.cross(tvec,edge1)
    v = np.dot(dir,qvec) * inv_det
    if v < 0 or u + v > 1.0:
        return NO_HIT

    z = np.dot(edge2,qvec) * inv_det
    return z


# @njit(parallel = True, cache = True) #nested lists are not supported in numba
def ids2names(z_ids : np.ndarray, solid_names : list[str]) -> list[list[str]]:
    '''
    input:
        z_ids - array of shape (n_rays, max_hits) containing meshes ids
                as provided in meshes_v and meshes_t
        solid_names - numpy array of strings to replace integers of z_ids

    output:
        outputs are ordered: closest hit to furtherst hit
        z_names
    '''

    out = [['' for _ in range(z_ids.shape[1])] for _ in range(z_ids.shape[0])]
    for i_r in range(z_ids.shape[0]):
        for i_h in range(z_ids.shape[1]):
            id = z_ids[i_r, i_h]
            out[i_r][i_h] = '' if id == NO_HIT else solid_names[id]
    return out


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
