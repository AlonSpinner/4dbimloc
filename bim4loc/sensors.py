from bim4loc.maps import Map, RayTracingMap
import numpy as np
from typing import Union
import open3d as o3d
from bim4loc.geometry import Pose2z

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

    def sense(self, pose : Pose2z, m : RayTracingMap) -> Union[np.ndarray,list[str]]:
        rays = o3d.core.Tensor([[pose.x,pose.y,pose.z,
                        np.cos(pose.theta+a),np.sin(pose.theta+a),0] for a in self.angles],
                        dtype=o3d.core.Dtype.Float32)

        ans = m._scene.cast_rays(rays)
        z = ans['t_hit'].numpy()
        solid_names = [m.solids[i].name if i != m._scene.INVALID_ID else '' for i in ans['geometry_ids'].numpy()]

        if self.std is not None:
            z = np.random.normal(z, self.std)

        z[z > self.max_range] = self.max_range

        return z, solid_names