import open3d as o3d
from bim4loc.binaries.paths import DRONE_PATH
from bim4loc.solid_objects import DynamicObject
from bim4loc.geometry import pose2

class Drone:
    def __init__(self, pose2 : pose2, hover_height = 0.0):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [1.0 , 0.0 , 0.0 , 1.0]
        base_drone_geo = o3d.io.read_triangle_mesh(DRONE_PATH)
        
        
        self.object = DynamicObject(name = 'drone', 
                                    geometry = base_drone_geo, 
                                    material = mat, 
                                    pose = pose2)
        self.pose2 = pose2
        self.hover_height = hover_height
        self.object.update_geometry(self.pose2, self.hover_height)

    def move(self, a):
        self.pose2 = self.pose2 + a
        self.object.update_geometry(self.pose2, self.hover_height)

    def scan(self):
        pass

