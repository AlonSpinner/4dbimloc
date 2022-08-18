from bim4loc.maps import Map, RayCastingMap
import open3d as o3d
from bim4loc.geometry.poses import Pose2z
import bim4loc.geometry.raytracer as raytracer
import numpy as np
from typing import Union

class Sensor():
    def __init__(self):
        pass

    def sense(self, pose : Pose2z) -> Union[np.ndarray,list[str]]: #to be overwritten
        '''
        returns np.ndarray of measurement values
        returns list of viewed solid names
        '''
        pass

    @staticmethod
    def forward_existence_model(z : str, m : str) -> float: #to be overwritten
        '''
        z - meaurement "⬜" or "⬛"
        m - cell state "⬜" or "⬛"
        
        returns probablity of achieving measurement z
        '''
        pass


class Lidar1D(Sensor):
    def __init__(self,
                angles : np.ndarray = np.linspace(-np.pi/2, np.pi/2, num = 36), 
                max_range : float = 10.0,
                std : float = None):
        
        self.angles = angles
        self.max_range = max_range
        self.std = std
        self.piercing = True

    def sense(self, pose : Pose2z, m : RayCastingMap, n_hits = 10):
        rays = self.get_rays(pose)

        if self.piercing == False: #not piercing, n_hits == 1 by definition. ignores input
            z_values, z_ids = raytracer.raytrace(rays, *m.scene, 1)
            z_names = raytracer.ids2names(z_ids, m.solid_names)
            
            z_values = z_values.squeeze()
            z_names = [zi_names[0] for zi_names in z_names]
        else:
            z_values, z_ids = raytracer.raytrace(rays, *m.scene, n_hits)
            z_names = raytracer.ids2names(z_ids, m.solid_names)
        
        z_values[z_values > self.max_range] = self.max_range
        # z_names[z_values > self.max_range] = '' #need to change later

        return z_values, z_names
            
    def get_rays(self, pose : Pose2z) -> np.ndarray:
        return np.vstack([(pose.x,pose.y,pose.z,
                        np.cos(pose.theta+a),np.sin(pose.theta+a),0) for a in self.angles])