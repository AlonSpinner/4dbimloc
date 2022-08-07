from tkinter import N
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

        def forward_measurement_model(self, pose : Pose2z, angles : np.ndarray, max_range : float) \
                                            -> Union[np.ndarray,np.ndarray]:
            rays = o3d.core.Tensor([[pose.x,pose.y,pose.z,
                                    np.cos(pose.theta+a),np.sin(pose.theta+a),0] for a in angles],
                                    dtype=o3d.core.Dtype.Float32)
            ans = self._scene.cast_rays(rays)
            z = ans['t_hit'].numpy()
            ids = ans['geometry_ids'].numpy()
            
            #anything above max_range will be consinderd as no-hit
            above_max_range = z > max_range
            ids[above_max_range] = -1
            
            z[z > max_range] = max_range
            return z, ids

        def bounds(self) -> Union[np.ndarray, np.ndarray]:
            all_points = np.vstack([np.asarray(o.geometry.vertices) for o in self._solids])
            all_points = o3d.utility.Vector3dVector(all_points)
            aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(all_points)
            return aabb.get_min_bound(), aabb.get_max_bound(), aabb.get_extent()






