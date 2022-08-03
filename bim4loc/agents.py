import open3d as o3d
import numpy as np
from bim4loc.binaries.paths import DRONE_PATH
from bim4loc.solids import DynamicSolid
from bim4loc.geometry import Pose2z
from bim4loc.maps import Map
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
        self.lidar_angles = np.linspace(-np.pi/2, np.pi/2, num = 36)
        self.lidar_max_range = 10.0
            
        self.solid.update_geometry(self.pose)

    def move(self, a : Pose2z, cov = None):
        if cov is not None:
            a = Pose2z(*np.random.multivariate_normal(a.Log(), cov))
            
        self.pose = self.pose.compose(a)
        self.solid.update_geometry(self.pose)

    def scan(self, m : Map, std = 0.1) -> Union[np.ndarray, np.ndarray]:
        z = m.forward_measurement_model(self.pose, self.lidar_angles, self.lidar_max_range)
        z = np.random.normal(z, std)
        
        world_thetas = (self.pose._theta + self.lidar_angles).reshape(-1,1)
        world_p = np.hstack((self.pose.x + z * np.cos(world_thetas), 
                       self.pose.y + z * np.sin(world_thetas),
                       self.pose.z * np.ones_like(z)))
        return z, world_p

