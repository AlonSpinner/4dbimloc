import open3d as o3d
import numpy as np
from bim4loc.binaries.paths import DRONE_PATH
from bim4loc.solids import DynamicSolid
from bim4loc.geometry.poses import Pose2z
from bim4loc.maps import RayCastingMap
from bim4loc.sensors import Lidar1D
from typing import Union

class Drone:
    def __init__(self, pose : Pose2z):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = [1.0 , 0.0 , 0.0 , 1.0]
        mat.base_roughness = 0.2
        mat.base_metallic = 1.0
        base_drone_geo = o3d.io.read_triangle_mesh(DRONE_PATH)
        self.solid = DynamicSolid(name = 'drone', 
                                    geometry = base_drone_geo, 
                                    material = mat, 
                                    pose = pose)
        
        
        self.pose = pose
        self.solid.update_geometry(self.pose)

    def mount_sensor(self, sensor : Lidar1D):
        #currently assume sensor pose is identical to agent pose
        self.sensor : Lidar1D = sensor

    def move(self, a : Pose2z, cov = None):
        if cov is not None:
            a = Pose2z(*np.random.multivariate_normal(a.Log(), cov))
            
        self.pose = self.pose.compose(a)
        self.solid.update_geometry(self.pose)

    def scan(self, m : RayCastingMap, project_scan = False) -> Union[np.ndarray, np.ndarray, list[str]]:
        '''
        output:
        z - 1D array
        world_p - MX3 matrix
        '''

        z, solid_names = self.sensor.sense(self.pose, m)
        
        if project_scan:
            drone_p = np.vstack((z * np.cos(self.sensor.angles), 
                        z * np.sin(self.sensor.angles),
                        np.zeros_like(z)))
            world_p = self.pose.transform_from(drone_p)
            return z, solid_names, world_p
        else:
            return z, solid_names

