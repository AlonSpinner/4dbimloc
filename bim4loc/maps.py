from tkinter import N
import numpy as np
import open3d as o3d
from bim4loc.geometry import Pose2z
from bim4loc.solids import IfcSolid
from typing import Union

class Map:
    def __init__(self, solids : list[IfcSolid]) -> None:
        self.solids : list[IfcSolid] = solids

    def bounds(self) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        all_points = np.vstack([np.asarray(o.geometry.vertices) for o in self.solids])
        all_points = o3d.utility.Vector3dVector(all_points)
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(all_points)
        return aabb.get_min_bound(), aabb.get_max_bound(), aabb.get_extent()

class BeliefMap(Map):
        def __init__(self, solids : list[IfcSolid]) -> None:
            super().__init__(solids)
            self.solid_names : list[str] = [s.name for s in solids]

        def update_belief(self, viewed_solid_names : list[str]) -> None:
            for name in viewed_solid_names:
                if name: #empty string relates to a sensor that provided "no hit" at the time step
                    self.solid_names.index(name)

class RayTracingMap(Map):
        def __init__(self, solids : list[IfcSolid]) -> None:
            super().__init__(solids)

            scene = o3d.t.geometry.RaycastingScene()
            for s in solids:
                s = o3d.t.geometry.TriangleMesh.from_legacy(s.geometry)
                scene.add_triangles(s)     
            self._scene =  scene

        def forward_measurement_model(self, pose : Pose2z, angles : np.ndarray, max_range : float) \
                                            -> Union[np.ndarray,list[str]]:
            rays = o3d.core.Tensor([[pose.x,pose.y,pose.z,
                                    np.cos(pose.theta+a),np.sin(pose.theta+a),0] for a in angles],
                                    dtype=o3d.core.Dtype.Float32)
            ans = self._scene.cast_rays(rays)
            z = ans['t_hit'].numpy()
            solid_names = [self.solids[i].name if i != self._scene.INVALID_ID else '' for i in ans['geometry_ids'].numpy()]
            
            z[z > max_range] = max_range
            return z, solid_names






