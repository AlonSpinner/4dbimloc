import numpy as np
import open3d as o3d
from bim4loc.geometry import pose2
from bim4loc.solid_objects import IfcObject
from typing import Union

class Map:
        def __init__(self, objects : list[IfcObject]) -> None:
            scene = o3d.t.geometry.RaycastingScene()
            for o in objects:
                o = o3d.t.geometry.TriangleMesh.from_legacy(o.geometry)
                scene.add_triangles(o)
            
            self._scene =  scene
            self._objects = objects

        def forward_measurement_model(self, pose2 : pose2, angles : np.ndarray, max_range : float) -> np.ndarray:
            rays = o3d.core.Tensor([[pose2.x,pose2.y,0,
                                    np.cos(pose2.theta+a),np.sin(pose2.theta+a),0] for a in angles],
                                    dtype=o3d.core.Dtype.Float32)
            ans = self._scene.cast_rays(rays)
            z = ans['t_hit'].numpy()
            z[z > max_range] = max_range
            return z.reshape(-1,1)

        def bounds(self) -> Union[np.ndarray, np.ndarray]:
            all_points = np.vstack([np.asarray(o.geometry.vertices) for o in self._objects])
            all_points = o3d.utility.Vector3dVector(all_points)
            aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(all_points)
            return aabb.get_min_bound(), aabb.get_max_bound()






