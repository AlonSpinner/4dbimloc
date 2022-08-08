import numpy as np
import open3d as o3d
from bim4loc.solids import IfcSolid
from typing import Union, Callable


class Map:
    def __init__(self, solids : list[IfcSolid]) -> None:
        self.solids : list[IfcSolid] = solids
        self.solid_names : list[str] = [s.name for s in solids]

    def bounds(self) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        all_points = np.vstack([np.asarray(o.geometry.vertices) for o in self.solids])
        all_points = o3d.utility.Vector3dVector(all_points)
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(all_points)
        return aabb.get_min_bound(), aabb.get_max_bound(), aabb.get_extent()
                
class RayTracingMap(Map):
        def __init__(self, solids : list[IfcSolid]) -> None:
            super().__init__(solids)

            scene = o3d.t.geometry.RaycastingScene()
            for s in solids:
                s = o3d.t.geometry.TriangleMesh.from_legacy(s.geometry)
                scene.add_triangles(s)     
            self._scene =  scene






