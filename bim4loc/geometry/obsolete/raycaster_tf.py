import numpy as np
from typing import Union
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #stops tf gibrish
import tensorflow as tf

EPS = tf.constant(1e-16, dtype = tf.float32)
NO_HIT = tf.constant(2161354.0,dtype = tf.float32)
INF_RANGE = 1e6

# @tf.function
def raycast(rays : tf.Tensor, meshes_v :  tf.Tensor, meshes_t :  tf.Tensor,
                    meshes_iguid :  tf.Tensor,
                    inc_v : int = 60, inc_t : int = 20,
                    max_hits : int = 10) -> Union[ tf.Tensor,  tf.Tensor]:
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

    z_values = tf.Variable(tf.fill((N_rays, max_hits), INF_RANGE))
    z_ids = tf.Variable(tf.fill((N_rays, max_hits), NO_HIT))
    z_normals = tf.Variable(tf.fill((N_rays, max_hits, 3), 0.0))

    for i_r in tf.range(N_rays):
        ray = rays[i_r]

        for i_m in tf.range(N_meshes):
            finished_mesh = False

            m_t = meshes_t[i_m * inc_t : (i_m + 1) * inc_t]
            m_v = meshes_v[i_m * inc_v : (i_m + 1) * inc_v]
            for i_t in tf.range(m_t.shape[0]):
                triangle = tf.gather(m_v,m_t[i_t])
                if tf.reduce_sum(triangle) == 0: #empty triangle
                    finished_mesh = True
                    break
                
                z, n = ray_triangle_intersection(ray, triangle)
                n = tf.reshape(n,(1,3))
                if z != NO_HIT and z > 0:
                    ii = tf.searchsorted(z_values[i_r], z)
                    if ii == max_hits: #new z is the biggest, throw it away
                        pass
                    elif ii == 0:
                        z_values[i_r].assign(tf.stack((tf.constant([z]), z_values[i_r][:-1])))
                        z_ids[i_r].assign(tf.stack((tf.constant([meshes_iguid[i_m]]), z_ids[i_r][:-1])))
                        z_normals[i_r].assign(tf.stack((n, z_normals[i_r][ii:-1])))
                    else:
                        z_values[i_r].assign(tf.stack((z_values[i_r][:ii], tf.constant([z]), z_values[i_r][ii:-1])))
                        z_ids[i_r].assign(tf.stack((z_ids[i_r][:ii], tf.constant([meshes_iguid[i_m]]), z_ids[i_r][ii:-1])))
                        z_normals[i_r].assign(tf.stack((z_normals[i_r][:ii,:], n, z_normals[i_r][ii:-1,:])))
            
            if finished_mesh:
                break
    
    return z_values, z_ids, z_normals

@tf.function
def ray_triangle_intersection(ray : tf.Tensor, triangle : tf.Tensor):
    '''
    based on https://github.com/substack/ray-triangle-intersection/blob/master/index.js

    input:
        ray ~ np.array([x,y,z,vx,vy,vz])
        triangle ~ np.array([[ax,ay,az],
                            [bx,by,bz],
                            [cx,cy,cz]])

    output:
        z - distance to intersection             
    '''
    eye = ray[:3]
    dir = ray[3:]
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    n = tf.zeros(3, dtype = tf.float32)

    pvec = tf.linalg.cross(dir,edge2)
    det = tf.tensordot(edge1,pvec,1)
    if det < EPS: #abs(det) < EPS if we want internal intersection
        return NO_HIT, n
    
    inv_det = 1.0/det
    tvec  = eye - triangle[0]
    u = tf.tensordot(tvec, pvec,1) * inv_det
    if u < 0.0 or u > 1.0:
        return NO_HIT, n
    qvec = tf.linalg.cross(tvec,edge1)
    v = tf.tensordot(dir,qvec,1) * inv_det
    if v < 0.0 or u + v > 1.0:
        return NO_HIT, n

    z = tf.tensordot(edge2,qvec,1) * inv_det
    n = tf.linalg.cross(edge1,edge2)
    n = n/tf.linalg.norm(n)
    return z, n

if __name__ == "__main__":
    #simple test to show functionality and speed
    import time
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print(f"device name: '{device_name}'")
        raise SystemError('GPU device not found')

    with tf.device(device_name):
        ray = tf.constant([0,0,0,1,0,0], dtype = tf.float32)
        triangle = tf.constant([[2,-1,-1],
                                [2,0,1],
                                [2,1,-1]], dtype = tf.float32)
        
        N = int(1e4)
        ray_triangle_intersection(ray, triangle)
        s = time.time()
        for _ in range(N):
            ray_triangle_intersection(ray, triangle)
        e = time.time()
        print((e-s)/N)
        print(ray.device)

        
        # ray = tf.reshape(ray, (1,6))
        # N = int(1e4)
        # raycast(ray, triangle,tf.constant([[0,1,2]]), tf.constant([0]), 3, 1, 1)
        # print('started')
        # s = time.time()
        # for _ in range(N):
        #     raycast(ray, triangle,tf.constant([[0,1,2]]), tf.constant([0]), 3, 1, 1)
        # e = time.time()
        # print((e-s)/N)