from dataclasses import dataclass
import ifcopenshell, ifcopenshell.geom
import numpy as np
import open3d as o3d
from open3d.visualization import rendering
import bim4loc.random.one_dim as r_1d
import bim4loc.random.utils as r_utils
from bim4loc.geometry.poses import Pose2z
from importlib import import_module
from copy import deepcopy

@dataclass(frozen = False)
class o3dSolid:
    name : str #name
    geometry : float #o3d.cuda.pybind.geometry.TriangleMesh
    material : float #o3d.cuda.pybind.visualization.rendering.MaterialRecord
    _min_alpha = 0.3

    def update_alpha(self, alpha : float) -> None:
        self.material.base_color = np.hstack((self.material.base_color[:3], max(alpha,self._min_alpha)))

@dataclass()
class IfcSolid(o3dSolid):
    schedule : r_1d.Distribution1D
    completion_time : float = 0.0
    ifc_color : np.ndarray = np.array([0, 0, 0])
    existance_belief : float = 0.0
    logOdds_existence_belief : float = 0.0
    
    def set_random_completion_time(self) -> None:
        s = self.schedule.sample()
        if s:
            self.completion_time = s[0]

    def is_complete(self, time : float) -> bool:
        return (time > self.completion_time)
    
    def set_existance_belief_by_schedule(self, time : float, set_shader = False) -> None:
        self.existance_belief = self.schedule.cdf(time)
        self.logOdds_existence_belief = r_utils.p2logodds(self.existance_belief)
        if set_shader:
            self.material.base_color = np.array([1, 0, 0, self.existance_belief])

    def set_existance_belief_and_shader(self, belief : float) -> None:
        self.existance_belief = belief #probablity
        self.logOdds_existence_belief = r_utils.p2logodds(self.existance_belief)
        self.material.base_color = np.array([1, 0, 0, belief])

    def clone(self) -> 'IfcSolid':
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLitTransparency" #if transparent, use "defaultLitTransparency": https://github.com/isl-org/Open3D/issues/2890
        mat.base_color = self.material.base_color.copy()

        mesh = o3d.geometry.TriangleMesh(vertices  = self.geometry.vertices, triangles = self.geometry.triangles)
        mesh.compute_triangle_normals()
        
        return IfcSolid(
            name = self.name,
            geometry = mesh,
            material = mat,
            schedule = deepcopy(self.schedule),
            ifc_color = self.ifc_color.copy(),                                
            )

class PcdSolid(o3dSolid):
    def __init__(self, pcd : np.ndarray = None):
        self.name = 'pcd'
        
        if pcd is None:
            pcd = [[0,0,0]]
            self.geometry = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 10.0
        mat.base_color = [1.0, 0.8, 0.0, 1.0]
        self.material = mat

    def update(self, pcd : np.ndarray) -> None:
        '''
        input:
        pcd - 3Xm matrix
        '''
        self.geometry.points = o3d.utility.Vector3dVector(pcd)

class LinesSolid(o3dSolid):
    def __init__(self, pts : np.ndarray = None, indicies : np.ndarray = None):
        self.name = 'lines'
        
        if pts is None or indicies is None:
            p0 = [[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0]]
            l0 = [[0,1]]
            c0 = [[1.0, 0.8, 0.0]]
            self.geometry = o3d.geometry.LineSet()
            self.geometry.points = o3d.utility.Vector3dVector(p0)
            self.geometry.lines = o3d.utility.Vector2iVector(l0)
            self.geometry.colors = o3d.utility.Vector3dVector(c0)

        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 10 
        self.material = mat

    def update(self, pts : np.ndarray = None, indicies : np.ndarray = None) -> None:
        '''
        input:
        pcd - 3Xm matrix
        '''
        self.geometry.points = o3d.utility.Vector3dVector(pts)
        self.geometry.lines = o3d.utility.Vector2iVector(indicies)
        # self.geometry.colors = o3d.utility.Vector3dVector(indicies)


class DynamicSolid(o3dSolid):
    base_geometry : float #o3d.cuda.pybind.geometry.TriangleMesh
    pose : Pose2z = Pose2z.identity()
    
    def __init__(self, name, geometry, material, pose = None):
        self.name = name
        self.geometry = geometry
        self.material = material
        self.base_geometry = geometry
        self.pose = pose

        if pose is not None:
            self.update_geometry(pose)

    def update_geometry(self, pose : Pose2z) -> None:
        self.geometry = deepcopy(self.base_geometry).transform(pose.Exp())
        self.pose = pose

class ArrowSolid(DynamicSolid):
    def __init__(self, name, alpha : float, pose = None):
        self.name = name
        
        self.geometry = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius = 0.07, 
                                                                cone_radius = 0.12, 
                                                                cylinder_height = 0.5, 
                                                                cone_height = 0.4)
        self.geometry.rotate(o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz(np.array([0,np.pi/2,0])))
        self.geometry.compute_triangle_normals()
        self.base_geometry = self.geometry
        
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLitTransparency" #"defaultUnlit", usefull for debugging purposes
        mat.base_color = np.array([0.0, 0.0, 1.0, 1.0])
        self.material = mat
        self.update_alpha(alpha) #sets base_color
        
        self.pose = pose
    
        if pose is not None:
            self.update_geometry(pose)

#----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------- IFC CONVERTION ----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------


def description2schedule(description : str) -> r_1d.Distribution1D:
    if description:
        try:
            lst = description.split(" ")
            _dname = lst[0]
            _dparams = [int(num) for num in lst[1:]]
            _class = getattr(import_module(r_1d.__name__),_dname)
            instance = _class.__new__(_class)
            instance.__init__(*_dparams)
            return instance
        except:
            print(f'description does not fit any programmed schedule time distribution in {r_1d.__name__}')
    else:
        return r_1d.Distribution1D() #empty

def ifc_converter(ifc_path) -> list[IfcSolid]:
    '''
    converts ifc file to a list of ifcSolids
    '''
    ifc = ifcopenshell.open(ifc_path)

    products = ifc.by_type("IfcProduct")
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS,True)
    settings.set(settings.APPLY_DEFAULT_MATERIALS, True)

    solids = []
    for product in products:
        if product.is_a("IfcOpeningElement"): continue
        if product.Representation: #has shape
            shape = ifcopenshell.geom.create_shape(settings, inst=product)
            m = shape.geometry.materials
            ifc_color = np.array(m[0].diffuse)
            element = ifc.by_guid(shape.guid)
                      
            verts = shape.geometry.verts # X Y Z of vertices in flattened list e.g. [v1x, v1y, v1z, v2x, v2y, v2z, ...]
            verts =  o3d.utility.Vector3dVector(np.array(verts).reshape((-1,3)))
            
            faces = shape.geometry.faces  #Indices of vertices per triangle face e.g. [f1v1, f1v2, f1v3, f2v1, f2v2, f2v3, ...]
            faces = o3d.utility.Vector3iVector(np.array(faces).reshape((-1,3)))
            
            mesh = o3d.geometry.TriangleMesh(vertices  = verts, triangles = faces)
            mesh.compute_triangle_normals()
            
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency" #if transparent, use "defaultLitTransparency": https://github.com/isl-org/Open3D/issues/2890
            mat.base_color = np.hstack((ifc_color, 1.0))

            solids.append(IfcSolid(
                                name = element.GlobalId,
                                geometry = mesh,
                                material = mat,
                                schedule = description2schedule(element.Description),
                                ifc_color = ifc_color,                                
                                ))

    return solids
