from bim4loc.maps import RayCastingMap
import bim4loc.geometry.raycaster as raycaster
from bim4loc.geometry.pose2z import R_from_theta
import numpy as np
from functools import partial
from typing import Tuple
from numba import njit, prange


class Sensor():
    def __init__(self):
        pass

    # def sense(self, pose : np.ndarray, noisy = True): #to be overwritten
    #     '''
    #     returns np.ndarray of measurement values
    #     returns list of viewed solid names
    #     '''
    #     pass

class Lidar(Sensor):
    def __init__(self,
                angles_u : np.ndarray = np.linspace(-np.pi/2, np.pi/2, num = 36), 
                angles_v : np.ndarray = np.linspace(-np.pi/30, np.pi/30, num = 3), 
                max_range : float = 10.0,
                std : float = 0.0,
                bias : float = 0.0):
        
        self.max_range = max_range
        self.std = std
        self.bias = bias

        self._angles_u = angles_u
        self._angles_v = angles_v
        self._Nu = angles_u.size
        self._Nv = angles_v.size

        #build rays
        ray_dirs = np.zeros((self._Nu*self._Nv, 3))
        for i, u in enumerate(angles_u):
            for j, v in enumerate(angles_v):
                ray_dirs[i*self._Nv + j] = spherical_coordiantes(u,v)
        self.ray_dirs = ray_dirs

    def sense_piercing(self, pose : np.ndarray, m : RayCastingMap, n_hits = 10, noisy = True):
        z_values, z_ids, z_normals, z_cos_incident, z_n_hits = \
            _sense(m.scene, n_hits, noisy, 
            self.ray_dirs, 
            self.bias, self.std, self.max_range,
            pose)
        return z_values, z_ids, z_normals, z_cos_incident, z_n_hits

    def sense_nonpiercing(self, pose : np.ndarray, m : RayCastingMap, n_hits = 10, noisy = True):
        z_values, z_ids, z_normals, z_cos_incident, z_n_hits = \
            _sense(m.scene, n_hits, noisy, 
            self.ray_dirs, 
            self.bias, self.std, self.max_range,
            pose)

        z_values = z_values[:,0]
        z_ids = z_ids[:,0]
        z_normals = z_normals[:,0]
        z_cos_incident = z_cos_incident[:,0]
        z_n_hits = np.minimum(z_n_hits,1)

        return z_values, z_ids, z_normals, z_cos_incident, z_n_hits

    def get_sense_piercing(self, m : RayCastingMap, n_hits = 10, noisy = True):
        #returns a njit function sense(x) that takes a pose x and returns the measurements

        #unpack variables to basic types for numba
        m_scene = m.scene
        ray_dirs = self.ray_dirs
        bias = self.bias
        std = self.std
        max_range = self.max_range
   
        # return partial(_sense,#numba doesnt work with partials..
        #                m_scene, n_hits, noisy, 
        #                ray_dirs, 
        #                bias, std, max_range)

        return lambda pose: _sense(m_scene, n_hits, noisy, 
                       ray_dirs, 
                       bias, std, max_range,
                       pose)

    def scan_to_points(self, z):
        return _scan_to_points(self._angles_u, self._angles_v, z)

    def get_scan_to_points(self):
        return partial(_scan_to_points, self._angles_u, self._angles_v)

####--------------------------------------------------------------------####
####--------------------------HELPING FUNCTIONS-------------------------####
####--------------------------------------------------------------------####


@njit(cache = True)
def ij2n(i: int , j: int , Nv: int) -> int:
    '''
    matrix indicies -> linear index 
    where Nv is the amount of columns
    '''
    return i*Nv + j

@njit(cache = True)
def n2ij(n : int, Nv : int ) -> tuple[int,int]:
    '''
    linear index  -> matrix indicies
    where Nv is the amount of columns
    '''
    return n//Nv, n%Nv

@njit(cache = True)
def spherical_coordiantes(u : float, v: float) -> np.ndarray:
    #can proably also work on np arrays u,v ?
    return np.array([np.cos(u)*np.cos(v),
                        np.sin(u)*np.cos(v), 
                        np.sin(v)])

# @njit(cache = True)
def _sense(m_scene : Tuple, n_hits : int, noisy : bool,
            ray_dirs : np.ndarray,
            bias : float , std: float, max_range : float,
            pose : np.ndarray):

    rays = transform_rays(pose, ray_dirs)

    z_values, z_ids, z_normals, z_cos_incident, z_n_hits = \
    raycaster.raycast(rays, 
                        m_scene[0], m_scene[1], m_scene[2], #numba doesnt know tuple unpacking
                        m_scene[3], m_scene[4], m_scene[5],
                        n_hits)

    if noisy:
        z_values = _noisify_measurements(z_values, bias, std)

    z_values = _cut_measurements(z_values, max_range)

    return z_values, z_ids, z_normals, z_cos_incident, z_n_hits

@njit(parallel = True, cache = True)
def transform_rays(pose, ray_dirs):
    '''
    input:
        pose - np.ndarray [x,y,z,theta]
        ray_dirs - np.ndarray [N_rays, 3] {yaw,pitch,roll}

    output:
        rays - np.ndarray [N_rays, 6] {x,y,z,dx,dy,dz}
    '''
    rays = np.zeros((ray_dirs.shape[0],6))
    rays[:, 0:3] = pose[:3]
    rays[:, 3:6] = (R_from_theta(pose[3]) @ ray_dirs.T).T
    return rays

@njit(parallel = True, cache = True)
def _noisify_measurements(z_values, bias, std):
    m = z_values[0].shape[0]
    n = z_values.shape[0]
    for i in prange(n):
        for j in prange(m):
            if np.isinf(z_values[i,j]):
                break
            z_values[i,j] = z_values[i,j] + np.random.normal(bias, std)
    return z_values

@njit(parallel = True, cache = True)
def _cut_measurements(z_values, max_range):
    m = z_values[0].shape[0]
    n = z_values.shape[0]
    for i in prange(n):
        for j in prange(m):
            if np.isinf(z_values[i,j]):
                z_values[i,j:] = max_range
                break
            z_values[i,j] = min(z_values[i,j], max_range)
    return z_values

@njit(parallel = True, cache = True)
def _scan_to_points(angles_u, angles_v, z):
    '''
    returns points in sensor frame
    if z.ndim == 2, the scan is from a simulation with piercing)
            shape ~ (N_rays, N_maxhits)

    returns points in sensor frame of size 3x N_rays * N_maxhits
    '''
    assert(angles_u.size * angles_v.size == z.shape[0])

    Nv = angles_v.size
    if z.ndim == 2:
        n = z.shape[0]
        m = z.shape[1]
        qz = np.zeros((3, n * m))

        for n_i in range(n):
                i,j = n2ij(n_i, Nv)
                qz[:, n_i * m : (n_i + 1) * m] = z[n_i,:] * spherical_coordiantes(angles_u[i], angles_v[j]).reshape(3,1)
        return qz
    else:
        n = z.shape[0] #amount of lasers in scan
        qz = np.zeros((3, n))
        for n_i in range(n):
                i,j = n2ij(n_i, Nv)
                qz[:, n_i] = z[n_i] * spherical_coordiantes(angles_u[i], angles_v[j])
        return qz