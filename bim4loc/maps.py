import numpy as np
import open3d as o3d
from bim4loc.solids import IfcSolid
from typing import Union, Callable


class Map:
    def __init__(self, solids_list : list[IfcSolid]) -> None:
        self.solid_names = [s.name for s in solids_list] #can extract later from dictionary but we will need it constantly
        self.solids : dict[IfcSolid] = dict(zip(self.solid_names,solids_list))

    def bounds(self) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        all_points = np.vstack([np.asarray(o.geometry.vertices) for o in self.solids.values()])
        all_points = o3d.utility.Vector3dVector(all_points)
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(all_points)
        return aabb.get_min_bound(), aabb.get_max_bound(), aabb.get_extent()
                
class RayTracingMap(Map):
        def __init__(self, solids_list : list[IfcSolid]) -> None:
            super().__init__(solids_list)

            scene = o3d.t.geometry.RaycastingScene()
            for s in self.solids.values():
                g = o3d.t.geometry.TriangleMesh.from_legacy(s.geometry)
                scene.add_triangles(g)     
            self._scene =  scene






