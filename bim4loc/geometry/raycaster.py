import numpy as np
from numba import njit, prange
from typing import Union
from bim4loc.geometry.minimal_distance import minimal_distance_from_projected_boundry

EPS = 1e-16
NO_HIT = 2161354

@njit(parallel = True, cache = True)
def raycast(rays : np.ndarray, meshes_v : np.ndarray, meshes_t : np.ndarray, meshes_bb : np.ndarray,
                    meshes_iguid : np.ndarray,
                    inc_v : int = 60, inc_t : int = 20,
                    max_hits : int = 10) -> Union[np.ndarray, np.ndarray]:
    '''
    input:
        rays - array of shape (n_rays, 6) containing [origin,direction]
        meshes_v - array of shape (n_meshes * inc_v, 3) containing vertices [px,py,pz]
        meshes_t - array of shape (n_meshes * inc_t, 3) containing triangles [id1,id2,id3]
        meshes_bb - arrya of shape (n_meshes * 6) containing AABB [minx,miny,minz,maxx,maxy,maxz]
        meshes_iguid - array of shape (n_meshes, ) containing mesh ids
        max_hits - after max_hits stop raytracing for ray.
                        this is an assumption that allows us to allocate size

        to clarify: 
            each mesh is contained of inc_v vertices and inc_t triangles
            if a triangle is [0,0,0] we will ignore it

    output:
        z_values - array of shape (n_rays, max_hits) containing range values
        z_ids - array of shape (n_rays, max_hits) containing meshes ids
                as provided in meshes_v and meshes_t
        z_normals - array of shape (n_rays, max_hits, 3) containing normals
        z_cos_incident - array of shape (n_rays, max_hits) containing cos of incident angle
        z_d_surface - array of shape (N_rays, max_hits) containing distance to surface
        
        outputs are sorted by z_values (small to big)
    '''
    N_meshes = int(meshes_t.shape[0]/ inc_t)
    N_rays = rays.shape[0]

    z_values = np.full((N_rays, max_hits), np.inf, dtype = np.float64)
    z_ids = np.full((N_rays, max_hits), NO_HIT, dtype = np.int32)
    z_normals = np.full((N_rays, max_hits, 3), 0.0 , dtype = np.float64)
    z_cos_incident = np.full((N_rays, max_hits), 0.0 , dtype = np.float64)
    z_d_surface = np.full((N_rays, max_hits), -1.0 , dtype = np.float64)

    for i_r in prange(N_rays):
        ray = rays[i_r]
        ray_dir = ray[3:]
        for i_m in prange(N_meshes):
            if not ray_box_intersection(ray[:3], ray_dir, meshes_bb[i_m]):
                continue
            
            finished_mesh = False
            m_t = meshes_t[i_m * inc_t : (i_m + 1) * inc_t]
            m_v = meshes_v[i_m * inc_v : (i_m + 1) * inc_v]
            for i_t in prange(m_t.shape[0]):
                triangle = m_v[m_t[i_t]]
                
                #DOES NOT ACCELERATE COMPUTATIONS?
                # if not(ray_box_intersection(ray[:3], ray_dir, triangle_to_AABB(triangle))):
                #     continue

                if triangle.sum() == 0: #empty triangle
                    finished_mesh = True
                    break
                
                z, n, c = ray_triangle_intersection(ray, triangle)
                n = n.reshape((1,3))
                if z != NO_HIT and z > 0:
                    ii = np.searchsorted(z_values[i_r], z)
                    if ii == max_hits: #new z is the biggest, throw it away
                        pass
                    elif ii == 0:
                        z_values[i_r] = np.hstack((np.array([z]), z_values[i_r][:-1]))
                        z_ids[i_r] = np.hstack((np.array([meshes_iguid[i_m]]), z_ids[i_r][:-1]))
                        z_normals[i_r] = np.vstack((n, z_normals[i_r][ii:-1]))
                        z_cos_incident[i_r] = np.hstack((np.array([c]), z_cos_incident[i_r][:-1]))
                    else:
                        z_values[i_r] = np.hstack((z_values[i_r][:ii], np.array([z]), z_values[i_r][ii:-1]))
                        z_ids[i_r] = np.hstack((z_ids[i_r][:ii], np.array([meshes_iguid[i_m]]), z_ids[i_r][ii:-1]))
                        z_normals[i_r] = np.vstack((z_normals[i_r][:ii,:], n, z_normals[i_r][ii:-1,:]))
                        z_cos_incident[i_r] = np.hstack((z_cos_incident[i_r][:ii], np.array([c]), z_cos_incident[i_r][ii:-1]))
            
            if finished_mesh:
                break
    
    return z_values, z_ids, z_normals, z_cos_incident, z_d_surface

@njit(fastmath = True, cache = True)
def ray_triangle_intersection(ray : np.ndarray, triangle : np.ndarray):
    '''
    based on https://github.com/substack/ray-triangle-intersection/blob/master/index.js

    input:
        ray - np.array([x,y,z,vx,vy,vz])
        triangle - np.array([[ax,ay,az],
                            [bx,by,bz],
                            [cx,cy,cz]])

    output:
        z - distance to intersection
        n - normal of triangle at intersection
        c - cosine of incident angle         
    '''
    eye = ray[:3]
    dir = ray[3:]
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    n = np.zeros(3)
    c = 0.0

    pvec = np.cross(dir,edge2)
    det = np.dot(edge1,pvec)
    if det < EPS: #abs(det) < EPS if we want internal intersection
        return NO_HIT, n, c
    
    inv_det = 1.0/det
    tvec  = eye - triangle[0]
    u = np.dot(tvec, pvec) * inv_det
    if u < 0 or u > 1.0:
        return NO_HIT, n, c
    qvec = np.cross(tvec,edge1)
    v = np.dot(dir,qvec) * inv_det
    if v < 0 or u + v > 1.0:
        return NO_HIT, n, c

    z = np.dot(edge2,qvec) * inv_det
    n = np.cross(edge1,edge2)
    n = n/np.linalg.norm(n)
    c = np.dot(n, dir)
    return z, n, c

@njit(fastmath = True, cache = True)
def triangle_to_AABB(triangle : np.ndarray) -> np.ndarray:
    '''
    input:
        triangle - np.array([[ax,ay,az],
                            [bx,by,bz],
                            [cx,cy,cz]])
    output:
        AABB - np.array([minx,miny,minz,maxx,maxy,maxz])
    '''
    vmin = np.minimum(np.minimum(triangle[0], triangle[1]),triangle[2])
    vmax = np.maximum(np.maximum(triangle[0], triangle[1]), triangle[2])
    return np.hstack((vmin, vmax))

@njit(fastmath = True, cache = True)
def ray_box_intersection(ray_o : np.ndarray, ray_dir : np.ndarray, box : np.ndarray) -> bool:
    '''
    based on https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525

    input:
        ray_o - np.array([x,y,z])
        ray_dir - np.array([dx,dy,dz])
        box - np.array([minx,miny,minz,maxx,maxy,maxz])

    output:
        boolean - True if ray intersects box          
    '''
    inv_ray_dir = 1/ray_dir #infs are acceptable
    t0s = (box[:3] - ray_o) * inv_ray_dir
    t1s = (box[3:] - ray_o) * inv_ray_dir

    t_bigger = np.maximum(t1s, t0s) #second hit

    #check the worst case
    tmax = np.min(t_bigger)
    if tmax < 0:
        return False
    else:
        return True

if __name__ == "__main__":
    #simple test to show functionality and speed
    import time
    ray = np.array([0,0,0,1,0,0],dtype=float)
    triangle = np.array([[2,-1,-1],
                        [2,0,1],
                        [2,1,-1]], dtype = float)
    
    # N = int(1e6)
    # ray_triangle_intersection(ray, triangle)
    # s = time.time()
    # for _ in range(N):
    #     ray_triangle_intersection(ray, triangle)
    # e = time.time()
    # print((e-s)/N)

    ray = ray.reshape((1,6))
    raycast(ray,triangle,np.array([[0,1,2]]),np.array([[2,-1,-1,2,1,1]]), np.array([0]), 3, 1, 5)
    N = int(1e2)    
    print('started')
    s = time.time()
    for _ in range(N):
        raycast(ray,triangle,np.array([[0,1,2]]),np.array([[2,-1,-1,2,1,1]]), np.array([0]), 3, 1, 1)
    e = time.time()
    print((e-s)/N)
    
    # raycast.parallel_diagnostics()
