import numpy as np
import open3d as o3d
from bim4loc.geometry import Pose2z
from bim4loc.solids import IfcSolid
from typing import Union

class Map:
        def __init__(self, solids : list[IfcSolid]) -> None:
            scene = o3d.t.geometry.RaycastingScene()
            for s in solids:
                s = o3d.t.geometry.TriangleMesh.from_legacy(s.geometry)
                scene.add_triangles(s)
            
            self._scene =  scene
            self._solids = solids

        def forward_measurement_model(self, pose : Pose2z, angles : np.ndarray, max_range : float) -> np.ndarray:
            rays = o3d.core.Tensor([[pose._x,pose._y,pose._z,
                                    np.cos(pose._theta+a),np.sin(pose._theta+a),0] for a in angles],
                                    dtype=o3d.core.Dtype.Float32)
            ans = self._scene.cast_rays(rays)
            z = ans['t_hit'].numpy()
            z[z > max_range] = max_range
            return z.reshape(-1,1)

        def bounds(self) -> Union[np.ndarray, np.ndarray]:
            all_points = np.vstack([np.asarray(o.geometry.vertices) for o in self._solids])
            all_points = o3d.utility.Vector3dVector(all_points)
            aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(all_points)
            return aabb.get_min_bound(), aabb.get_max_bound()






