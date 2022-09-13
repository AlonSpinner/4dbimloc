import open3d as o3d
import numpy as np
from bim4loc.binaries.paths import DRONE_PATH
from bim4loc.solids import DynamicSolid
from bim4loc.maps import RayCastingMap
from bim4loc.sensors import Lidar
from typing import Union
from bim4loc.geometry.pose2z import compose_s, transform_from

class Drone:
    def __init__(self, pose : np.ndarray):
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

    def mount_sensor(self, sensor : Lidar):
        #currently assume sensor pose is identical to agent pose
        self.sensor : Lidar = sensor

    def move(self, u : np.ndarray, cov : np.ndarray = None):
        
        if cov is not None:
            u = np.random.multivariate_normal(u, cov)
        
        self.pose = compose_s(self.pose, u)
        self.solid.update_geometry(self.pose)

    def scan(self, m : RayCastingMap, project_scan = False, noisy = True) -> Union[np.ndarray, np.ndarray, list[str]]:
        '''
        output:
        z - 1D array
        world_p - MX3 matrix
        '''

        z, z_ids, z_normals = self.sensor.sense(self.pose, m, noisy = noisy)
        
        if project_scan:
            drone_p = self.sensor.scan_to_points(z)
            world_p = transform_from(self.pose, drone_p)
            return z, z_ids, z_normals, world_p
        else:
            return z, z_ids, z_normals

