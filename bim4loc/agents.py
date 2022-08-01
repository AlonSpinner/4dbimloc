import open3d as o3d
import numpy as np
from bim4loc.binaries.paths import DRONE_PATH
from bim4loc.solid_objects import DynamicObject
from bim4loc.geometry import pose2
from bim4loc.maps import Map

class Drone:
    def __init__(self, pose2 : pose2, hover_height = 0.0):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [1.0 , 0.0 , 0.0 , 1.0]
        mat.base_roughness = 0.2
        mat.base_metallic = 1.0
        base_drone_geo = o3d.io.read_triangle_mesh(DRONE_PATH)
        self.object = DynamicObject(name = 'drone', 
                                    geometry = base_drone_geo, 
                                    material = mat, 
                                    pose = pose2)
        
        
        self.pose2 = pose2
        self.hover_height = hover_height
        self.lidar_angles = np.linspace(-np.pi/2, np.pi/2, num = 60)
        self.lidar_max_range = 10.0
            
        self.object.update_geometry(self.pose2, self.hover_height)

    def move(self, a : pose2, cov = None):
        if cov is not None:
            a = pose2(*np.random.multivariate_normal(a.local(), cov))
            
        self.pose2 = self.pose2 + a
        self.object.update_geometry(self.pose2, self.hover_height)

    def scan(self, m : Map, std = None):
        z = m.forward_measurement_model(self.pose2, self.lidar_angles, self.lidar_max_range)
        if std is not None:
            z = np.random.normal(z, std)
        
        world_thetas = (self.pose2.theta + self.lidar_angles).reshape(-1,1)
        world_p = np.hstack((self.pose2.x + z * np.cos(world_thetas), 
                       self.pose2.y + z * np.sin(world_thetas),
                       self.hover_height * np.ones_like(z)))
        return z, world_p

