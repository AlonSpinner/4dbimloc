from bim4loc.maps import RayCastingMap
import bim4loc.geometry.raycaster as raycaster
from bim4loc.geometry.pose2z import R_from_theta
import numpy as np
from functools import partial
from typing import Tuple
from numba import njit

class Sensor():
    def __init__(self):
        pass

    def sense(self, pose : np.ndarray, noisy = True): #to be overwritten
        '''
        returns np.ndarray of measurement values
        returns list of viewed solid names
        '''
        pass

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
        self.piercing = True

        self._angles_u = angles_u
        self._angles_v = angles_v
        self._Nu = angles_u.size
        self._Nv = angles_v.size

        #build rays
        rays = np.zeros((self._Nu*self._Nv, 6))
        for i, u in enumerate(angles_u):
            for j, v in enumerate(angles_v):
                rays[i*self._Nv + j, 3:6] = spherical_coordiantes(u,v)
        self._rays = rays


    def sense(self, pose : np.ndarray, m : RayCastingMap, n_hits = 10, noisy = True):
        return self._sense(pose, m.scene, n_hits, noisy, 
                            self._rays, self.piercing, 
                            self.bias, self.std, self.max_range)
    
    def get_sense(self):
        return partial(self._sense, 
                        self._rays, self.piercing, 
                        self.bias, self.std, self.max_range)
    
    def scan_to_points(self, z):
        return self._scan_to_points(self._angles_u, self._angles_v, z)

    def get_scan_to_points(self):
        return partial(self._scan_to_points, self._angles_u, self._angles_v)

    @staticmethod
    # @njit(cache = True)
    def _sense(pose : np.ndarray, m_scene : Tuple, n_hits : int, noisy : bool,
               ego_rays : np.ndarray,  piercing : bool,
               bias : float , std: float, max_range : float):
        rays = transform_rays(pose, ego_rays)
        
        z_values, z_ids, z_normals, z_cos_incident, z_n_hits = \
            raycaster.raycast(rays, 
                              m_scene[0], m_scene[1], m_scene[2], 
                              m_scene[3], m_scene[4], m_scene[5],
                              n_hits)
        
        if piercing == False:
            z_values = z_values[:,0]
            z_ids = z_ids[:,0]
            z_normals = z_normals[:,0]
            z_cos_incident = z_cos_incident[:,0]
            z_n_hits = np.minimum(z_n_hits,1)
        
        if noisy:
            z_values = np.random.normal(z_values + bias, std)

        z_values[z_values > max_range] = max_range

        return z_values, z_ids, z_normals, z_cos_incident, z_n_hits

    @staticmethod
    @njit(cache = True)
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

@njit(cache = True)
def transform_rays(pose, ego_rays):
    '''
    pose - np.ndarray [x,y,z,theta]
    ego_rays - np.ndarray [N_rays, 6] {x,y,z,yaw,pitch,roll}
    '''
    rays = ego_rays.copy()
    dirs = ego_rays[:,3:6].T
    rays[:, 0:3] = pose[:3]
    rays[:, 3:6] = (R_from_theta(pose[3]) @ dirs).T
    return rays