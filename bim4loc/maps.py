import numpy as np
import open3d as o3d
from bim4loc.solids import IfcSolid
from typing import Union
from collections import namedtuple

class Map:
    def __init__(self, solids : list[IfcSolid]) -> None:
        self.solids : list[IfcSolid] = solids #ordered array of solids!

    def update_solids_beliefs(self, beliefs) -> None:
        #this is only for visuals. beliefs are not stored inside the map object!
        for ii, p in enumerate(beliefs):
            self.solids[ii].set_existance_belief_and_shader(p)    

    def bounds(self) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        all_points = np.vstack([np.asarray(o.geometry.vertices) for o in self.solids])
        all_points = o3d.utility.Vector3dVector(all_points)
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(all_points)
        return aabb.get_min_bound(), aabb.get_max_bound(), aabb.get_extent()
    
SceneType = namedtuple('scene', ['vertices', 
                                'triangles',
                                'bounding_boxes',
                                'iguids',
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
            meshes_bb - arrya of shape (n_meshes * 6) containing AABB [minx,miny,minz,maxx,maxy,maxz]
            meshes_iguid - array of shape (n_meshes, ) containing mesh ids
            inc_v - amounts of rows that contain a single mesh data in meshes_v
            inc_t - amounts of rows that contain a single mesh data in meshes_t

            to clarify: 
            each mesh is contained of inc_v vertices and inc_t triangles
            if a triangle is [0,0,0] we will ignore it (and understand we finished with the mesh)
        '''
        max_vertices = 0
        max_triangles = 0
        for s in self.solids:
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
        meshes_bb = np.zeros((n, 6), dtype = np.float64)
        meshes_iguid = np.zeros((n), dtype = np.int32)
        for i_s, s in enumerate(self.solids):
            v = np.asarray(s.geometry.vertices)
            t = np.asarray(s.geometry.triangles)
            meshes_v[i_s * inc_v : i_s * inc_v + v.shape[0]] = v
            meshes_t[i_s * inc_t : i_s * inc_t + t.shape[0]] = t
            meshes_bb[i_s] = np.hstack((s.geometry.get_min_bound(), s.geometry.get_max_bound()))
            meshes_iguid[i_s] = i_s

        return SceneType(meshes_v, meshes_t, meshes_bb, meshes_iguid ,inc_v, inc_t)




