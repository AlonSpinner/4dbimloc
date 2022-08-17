import numpy as np
import open3d as o3d
from bim4loc.solids import IfcSolid
from typing import Union
from collections import namedtuple

class Map:
    def __init__(self, solids_list : list[IfcSolid]) -> None:
        self.solid_names = [s.name for s in solids_list] #can extract later from dictionary but we will need it constantly
        self.solids : dict[IfcSolid] = dict(zip(self.solid_names,solids_list))

    def bounds(self) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        all_points = np.vstack([np.asarray(o.geometry.vertices) for o in self.solids.values()])
        all_points = o3d.utility.Vector3dVector(all_points)
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(all_points)
        return aabb.get_min_bound(), aabb.get_max_bound(), aabb.get_extent()
                

SceneType = namedtuple('scene', ['vertices', 
                                'triangles', 
                                'inc_v', 
                                'inc_t'])

class RayCastingMap(Map):
    def __init__(self, solids_list : list[IfcSolid]) -> None:
        super().__init__(solids_list)
        self.scene = self.scene_from_solids()

    def scene_from_solids(self) -> SceneType:   
        '''
        outputs:
            meshes_v - array of shape (n_meshes * inc_v, 3) containing vertices [px,py,pz]
            meshes_t - array of shape (n_meshes * inc_t, 3) containing triangles [id1,id2,id3]
            inc_v - amounts of rows that contain a single mesh data in meshes_v
            inc_t - amounts of rows that contain a single mesh data in meshes_t

            to clarify: 
            each mesh is contained of inc_v vertices and inc_t triangles
            if a triangle is [0,0,0] we will ignore it
        '''
        max_vertices = 0
        max_triangles = 0
        for s in self.solids.values():
            n_v = (np.asarray(s.geometry.vertices)).shape[0]
            n_t = (np.asarray(s.geometry.triangles)).shape[0]

            if n_v > max_vertices:
                max_vertices = n_v
            if n_t > max_triangles:
                max_triangles = n_t

        inc_v = max_vertices
        inc_t = max_triangles

        n = len(self.solids)
        meshes_v = np.zeros((n * inc_v,3), dtype = np.float64)
        meshes_t = np.zeros((n * inc_t,3), dtype = np.int32)
        for i_s, s in enumerate(self.solids.values()):
            v = np.asarray(s.geometry.vertices)
            t = np.asarray(s.geometry.triangles)
            meshes_v[i_s * inc_v : i_s * inc_v + v.shape[0]] = v
            meshes_t[i_s * inc_t : i_s * inc_t + t.shape[0]] = t

        return SceneType(meshes_v, meshes_t, inc_v, inc_t)




