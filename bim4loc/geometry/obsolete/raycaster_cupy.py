import cupy as cp
from typing import Union

EPS = 1e-16
NO_HIT = 2161354

def raycast(rays : cp.ndarray, meshes_v : cp.ndarray, meshes_t : cp.ndarray,
                    meshes_iguid : cp.ndarray,
                    inc_v : int = 60, inc_t : int = 20,
                    max_hits : int = 10) -> Union[cp.ndarray, cp.ndarray]:
    '''
    input:
        rays - array of shape (n_rays, 6) containing [origin,direction]
        meshes_v - array of shape (n_meshes * inc_v, 3) containing vertices [px,py,pz]
        meshes_t - array of shape (n_meshes * inc_t, 3) containing triangles [id1,id2,id3]
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
        outputs are sorted by z_values (small to big)
    '''
    N_meshes = int(meshes_t.shape[0]/ inc_t)
    N_rays = rays.shape[0]

    z_values = cp.full((N_rays, max_hits), cp.inf, dtype = cp.float64)
    z_ids = cp.full((N_rays, max_hits), NO_HIT, dtype = cp.int32)
    z_normals = cp.full((N_rays, max_hits, 3), 0.0 , dtype = cp.float64)

    for i_r in cp.arange(N_rays):
        ray = rays[i_r]

        for i_m in cp.arange(N_meshes):
            finished_mesh = False

            m_t = meshes_t[i_m * inc_t : (i_m + 1) * inc_t]
            m_v = meshes_v[i_m * inc_v : (i_m + 1) * inc_v]
            for i_t in cp.arange(m_t.shape[0]):
                triangle = m_v[m_t[i_t]]
                if triangle.sum() == 0: #empty triangle
                    finished_mesh = True
                    break
                
                z, n = ray_triangle_intersection(ray, triangle)
                n = n.reshape((1,3))
                if z != NO_HIT and z > 0:
                    ii = cp.searchsorted(z_values[i_r], cp.array([z]))
                    if ii == max_hits: #new z is the biggest, throw it away
                        pass
                    elif ii == 0:
                        z_values[i_r] = cp.hstack((cp.array([z]), z_values[i_r][:-1]))
                        z_ids[i_r] = cp.hstack((cp.array([meshes_iguid[i_m]]), z_ids[i_r][:-1]))
                        z_normals[i_r] = cp.vstack((n, z_normals[i_r][ii:-1]))
                    else:
                        z_values[i_r] = cp.hstack((z_values[i_r][:ii], cp.array([z]), z_values[i_r][ii:-1]))
                        z_ids[i_r] = cp.hstack((z_ids[i_r][:ii], cp.array([meshes_iguid[i_m]]), z_ids[i_r][ii:-1]))
                        z_normals[i_r] = cp.vstack((z_normals[i_r][:ii,:], n, z_normals[i_r][ii:-1,:]))
            
            if finished_mesh:
                break

    return z_values, z_ids, z_normals

def ray_triangle_intersection(ray : cp.ndarray, triangle : cp.ndarray):
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
    n = cp.zeros(3)

    pvec = cp.cross(dir,edge2)
    det = cp.dot(edge1,pvec)
    if det < EPS: #abs(det) < EPS if we want internal intersection
        return NO_HIT, n
    
    inv_det = 1.0/det
    tvec  = eye - triangle[0]
    u = cp.dot(tvec, pvec) * inv_det
    if u < 0 or u > 1.0:
        return NO_HIT, n
    qvec = cp.cross(tvec,edge1)
    v = cp.dot(dir,qvec) * inv_det
    if v < 0 or u + v > 1.0:
        return NO_HIT, n

    z = cp.dot(edge2,qvec) * inv_det
    n = cp.cross(edge1,edge2)
    n = n/cp.linalg.norm(n)
    return z, n

if __name__ == "__main__":
    #simple test to show functionality and speed
    import time
    ray = cp.array([0,0,0,1,0,0],dtype=float)
    triangle = cp.array([[2,-1,-1],
                        [2,0,1],
                        [2,1,-1]], dtype = float)
    
    # N = int(1e4)
    # ray_triangle_intersection(ray, triangle)
    # s = time.time()
    # for _ in range(N):
    #     ray_triangle_intersection(ray, triangle)
    # e = time.time()
    # print((e-s)/N)

    ray = ray.reshape((1,6))
    raycast(ray,triangle,cp.array([[0,1,2]]),cp.array([0]), 3, 1, 1)
    N = int(1e2)
    print('started')
    s = time.time()
    for _ in range(N):
        raycast(ray,triangle,cp.array([[0,1,2]]),cp.array([0]), 3, 1, 1)
    e = time.time()
    print((e-s)/N)
