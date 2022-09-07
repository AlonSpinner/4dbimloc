from bim4loc.maps import RayCastingMap
from bim4loc.geometry.poses import Pose2z
import bim4loc.geometry.raycaster as raycaster
import numpy as np
from functools import partial

class Sensor():
    def __init__(self):
        pass

    def sense(self, pose : Pose2z, noisy = True): #to be overwritten
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


    def sense(self, pose : Pose2z, m : RayCastingMap, n_hits = 10, noisy = True):
        rays = self.transform_rays(pose)
        
        z_values, z_ids, z_normals = raycaster.raycast(rays, *m.scene, n_hits)
        
        if self.piercing == False:
            z_values = z_values[:,0]
            z_normals = z_normals[:,0]
        
        if noisy:
            z_values = np.random.normal(z_values + self.bias, self.std)

        z_values[z_values > self.max_range] = self.max_range

        return z_values, z_ids, z_normals
    
    def transform_rays(self, pose : Pose2z) -> np.ndarray:
        #returns rays in world system to be used by raycaster.
        rays = self._rays.copy()
        rays[:, 0:3] = pose.t.T
        rays[:, 3:6] = (pose.R @ rays[:, 3:6].T).T
        return rays
    
    def scan_to_points(self, z):
        return self._scan_to_points(self._angles_u, self._angles_v, z)
    
    def get_scan_to_points(self):
        return partial(self._scan_to_points, self._angles_u, self._angles_v)

    @staticmethod
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


def ij2n(i: int , j: int , Nv: int) -> int:
    '''
    matrix indicies -> linear index 
    where Nv is the amount of columns
    '''
    return i*Nv + j

def n2ij(n : int, Nv : int ) -> tuple[int,int]:
    '''
    linear index  -> matrix indicies
    where Nv is the amount of columns
    '''
    return n//Nv, n%Nv

def spherical_coordiantes(u : float, v: float) -> np.ndarray:
    #can proably also work on np arrays u,v ?
    return np.array([np.cos(u)*np.cos(v),
                        np.sin(u)*np.cos(v), 
                        np.sin(v)])