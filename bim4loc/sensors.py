from bim4loc.maps import RayCastingMap
from bim4loc.geometry.poses import Pose2z
import bim4loc.geometry.raycaster as raycaster
import numpy as np
from typing import Union
from functools import partial

class Sensor():
    def __init__(self):
        pass

    def sense(self, pose : Pose2z) -> Union[np.ndarray,list[str]]: #to be overwritten
        '''
        returns np.ndarray of measurement values
        returns list of viewed solid names
        '''
        pass

class Lidar1D(Sensor):
    def __init__(self,
                angles : np.ndarray = np.linspace(-np.pi/2, np.pi/2, num = 36), 
                max_range : float = 10.0,
                std : float = 0.0,
                bias : float = 0.0):
        
        self.angles = angles
        self.max_range = max_range
        self.std = std
        self.bias = bias
        self.piercing = True

    def sense(self, pose : Pose2z, m : RayCastingMap, n_hits = 10, noisy = True):
        rays = self.get_rays(pose)
        
        z_values, z_ids = raycaster.raycast(rays, *m.scene, n_hits)
        
        if self.piercing == False: #not piercing, n_hits == 1 by definition. ignores input
            z_values = z_values[:,0]
        
        if noisy:
            z_values = np.random.normal(z_values + self.bias, self.std)

        z_values[z_values > self.max_range] = self.max_range

        return z_values, z_ids
    
    def get_rays(self, pose : Pose2z) -> np.ndarray:
        #returns rays to be used by raycaster.        
        return np.vstack([(pose.x,pose.y,pose.z,
                        np.cos(pose.theta+a),np.sin(pose.theta+a),0) for a in self.angles])
    
    @staticmethod
    def scan_to_points(angles, z):
        #returns points in sensor frame
        if z.ndim == 2: #layered, simulated scan (angles, n_hits)
            n = z.shape[0] #amount of lasers in scan
            m = z.shape[1] #amount of hits per laser
            pz = np.zeros((3, n * m))
            for k in range(z.shape[1]):
                 z_k = z[:,k]
                 pz[:, k * n : (k+1) * n] = np.vstack((z_k * np.cos(angles), 
                                                       z_k * np.sin(angles),
                                                       np.zeros_like(z_k)))
            return pz
        else:
            return np.vstack((z * np.cos(angles), 
                              z * np.sin(angles),
                              np.zeros_like(z)))

    def get_scan_to_points(self):
        return partial(self.scan_to_points, self.angles)