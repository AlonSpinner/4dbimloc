import numpy as np
import open3d as o3d
from bim4loc.geometry import pose2
from bim4loc.solid_objects import IfcObject

class Map:
        def __init__(self, objects : list[IfcObject]):
            scene = o3d.t.geometry.RaycastingScene()
            for o in objects:
                o = o3d.t.geometry.TriangleMesh.from_legacy(o.geometry)
                scene.add_triangles(o)
            
            self._scene =  scene

        def forward_measurement_model(self, pose2 : pose2, angles : np.ndarray):
            rays = o3d.core.Tensor([[pose2.x,pose2.y,0,
                                    np.cos(pose2.theta+a),np.sin(pose2.theta+a),0] for a in angles],
                                    dtype=o3d.core.Dtype.Float32)
            ans = self._scene.cast_rays(rays)
            z = ans['t_hit'].numpy()
            return z.reshape(-1,1)




