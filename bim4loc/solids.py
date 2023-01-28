from dataclasses import dataclass, field
import ifcopenshell, ifcopenshell.geom
import numpy as np
from numpy.polynomial import polyutils
import open3d as o3d
from open3d.visualization import rendering
import bim4loc.random.one_dim as r_1d
from bim4loc.geometry.pose2z import compose_s_array, T_from_s
from importlib import import_module
from copy import deepcopy
from typing import Literal
import yaml
import matplotlib.colors as colors
from matplotlib import cm


@dataclass(frozen = False)
class o3dSolid:
    name : str #name
    geometry : float #o3d.cuda.pybind.geometry.TriangleMesh
    material : float #o3d.cuda.pybind.visualization.rendering.MaterialRecord
    _min_alpha = 0.3

    def update_alpha(self, alpha : float) -> None:
        self.material.base_color = np.hstack((self.material.base_color[:3], max(alpha,self._min_alpha)))

    def get_vertices(self) -> np.ndarray:
        return np.asarray(self.geometry.vertices)
    
    def set_vertices(self, vertices : np.ndarray) -> None:
        '''
        vertices ~ mx3 array
        '''
        self.geometry.vertices = o3d.utility.Vector3dVector(vertices)

@dataclass()
class IfcSolid(o3dSolid):
    schedule : r_1d.Distribution1D
    completion_time : float = 0.0
    ifc_color : np.ndarray = np.array([0, 0, 0])
    ifc_type : str = ''
    existence_dependence : list[str] = field(default_factory=list)  #cant have mutables...
    
    def set_random_completion_time(self) -> None:
        s = self.schedule.sample()
        if s:
            self.completion_time = s[0]

    def is_complete(self, time : float) -> bool:
        return (time > self.completion_time)

    def set_existance_belief_and_shader(self, belief : float) -> None:
        if belief > 0.9:
            self.material.base_color = np.array([0, 1, 0, belief])
        else:
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
            ifc_type = self.ifc_type, #str is immutable
            existence_dependence = self.existence_dependence.copy()
            )

class Label3D():
    '''
    we use this for the visualizer like the rest of the solids, might aswell be here.
    '''
    def __init__(self, text: str, 
                       position : np.ndarray):
        self.text = text
        self.position = position

class PcdSolid(o3dSolid):
    def __init__(self, name : str = 'pcd', 
                    pcd : np.ndarray = None,
                    color = np.array([1.0, 0.8, 0.0]),
                    shader : Literal["defaultUnlit", "normals"] = "defaultUnlit"):
        
        '''
        pcd - m X 3
        '''
        
        self.name = name
        
        if pcd is None:
            pcd = [[100,100,100]] #random far off point 
        self.geometry = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))

        mat = rendering.MaterialRecord()
        mat.shader = shader
        mat.point_size = 10.0
        self.material = mat

        if color.ndim == 2: #if color is a matrix
            self.geometry.colors = o3d.utility.Vector3dVector(color)
        else:
            # self.geometry.paint_uniform_color(color) #<--- no effect?
            self.material.base_color = np.hstack((color,1.0))

    def update(self, pcd : np.ndarray, normals = None, color = None) -> None:
        '''
        input:
        pcd - mx3 matrix
        normals - mx3 matrix
        '''
        self.geometry.points = o3d.utility.Vector3dVector(pcd)
        if normals is not None:
            self.geometry.normals = o3d.utility.Vector3dVector(normals)

        if color is not None:
            if color.ndim == 2: #if color is a matrix
                self.geometry.colors = o3d.utility.Vector3dVector(color)
            else:
                self.geometry.paint_uniform_color(color)

class LinesSolid(o3dSolid):
    def __init__(self, name = 'lines',
                       pts : np.ndarray = None, 
                       indicies : np.ndarray = None,
                       line_width : float = 2.0,
                       color : np.ndarray = np.array([1.0, 0.8, 0.0])):
        '''
        pts - m X 3
        indicies - m X 2
        '''
        
        self.name = name
        self.color = color
        
        if pts is None or indicies is None:
            pts = [[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0]]
            indicies = np.array([[0,1]])

        self.geometry = o3d.geometry.LineSet()
        self.geometry.points = o3d.utility.Vector3dVector(pts)
        self.geometry.lines = o3d.utility.Vector2iVector(indicies)
        c = np.tile(self.color, (indicies.shape[0], 1))
        self.geometry.colors = o3d.utility.Vector3dVector(c)

        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = line_width 
        self.material = mat

    def update(self, pts : np.ndarray, indicies : np.ndarray) -> None:
        '''
        input:
        pcd - mx3 matrix
        '''
        self.geometry.points = o3d.utility.Vector3dVector(pts)
        self.geometry.lines = o3d.utility.Vector2iVector(indicies)
        c = np.tile(self.color, (indicies.shape[0], 1))
        self.geometry.colors = o3d.utility.Vector3dVector(c)

class TrailSolid(LinesSolid):
    def __init__(self, name, p0):
        super().__init__(name = name, pts = np.vstack((p0,p0+0.0001)), #points have to be different
                         indicies = np.array([[0,1]]), 
                         line_width = 5.0, 
                         color = np.array([0.0, 0.0, 0.0]))
    def update(self, pts) -> None:
        '''
        input:
        pts - mx3 matrix

        adds points to existing line
        '''
        old_points = np.asarray(self.geometry.points)
        self.geometry.points = o3d.utility.Vector3dVector(np.vstack((old_points, pts)))
        old_lines = np.asarray(self.geometry.lines)
        new_lines_left = old_lines[-1,1] + np.arange(0,pts.shape[0])
        new_lines_right = new_lines_left + 1
        new_lines = np.hstack((new_lines_left, new_lines_right))
        lines = np.vstack((old_lines, new_lines))
        self.geometry.lines = o3d.utility.Vector2iVector(lines)
        c = np.tile(self.color, (lines.shape[0], 1))
        self.geometry.colors = o3d.utility.Vector3dVector(c)                                                     

class ScanSolid(o3dSolid):
    def __init__(self, name = 'scan',
                p0 : np.ndarray = None,
                pts : np.ndarray = None, 
                line_width : float = 2.0,
                color : np.ndarray = np.array([1.0, 0.8, 0.0])):

        '''
        p0 -  1 x 3
        pts - m x 3
        '''

        self.name = name
        self.color = color

        if p0 is None or pts is None:
            p0 = np.array([0.0, 0.0, 0.0])
            pts = np.array([0.1, 0.0, 0.0])

        pts = np.vstack((p0, pts))
        indicies = np.zeros((pts.shape[0],2), dtype = int)
        indicies[:,1] = np.arange(pts.shape[0])
        
        self.geometry = o3d.geometry.LineSet()
        self.geometry.points = o3d.utility.Vector3dVector(pts)
        self.geometry.lines = o3d.utility.Vector2iVector(indicies)
        c = np.tile(self.color, (indicies.shape[0], 1))
        self.geometry.colors = o3d.utility.Vector3dVector(c)

        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = line_width 
        self.material = mat

    def update(self,  p0 : np.ndarray,
                        pts : np.ndarray):

        pts = np.vstack((p0, pts))
        indicies = np.zeros((pts.shape[0],2), dtype = int)
        indicies[:,1] = np.arange(pts.shape[0])
        
        self.geometry.points = o3d.utility.Vector3dVector(pts)
        self.geometry.lines = o3d.utility.Vector2iVector(indicies)
        c = np.tile(self.color, (indicies.shape[0], 1))
        self.geometry.colors = o3d.utility.Vector3dVector(c)

class ParticlesSolid(o3dSolid):
    def __init__(self, name = 'particles', 
                       poses : np.ndarray = None, 
                       scale = 0.4,
                       line_width = 4.0,
                       line_color : np.ndarray = np.array([0.0, 0.0, 0.0]),
                       use_weight_colors : bool = True,
                       tail_color : np.ndarray = np.array([0.0, 0.5, 0.5])):
        '''
        poses - mx4, [x,y,theta,z]
        '''
        self.name = name
        self.scale = scale
        self.use_weight_colors = use_weight_colors
        
        if poses is None:
            poses = np.zeros(4)

        ds = np.array([self.scale,0,0,0])
        heads = compose_s_array(poses,ds)[:,:3]
        tails = poses[:,:3]
        indicies =  np.vstack((np.arange(0,len(poses), dtype = int), #2xm
                                np.arange(len(poses),2 * len(poses), dtype = int)))

        self.lines = LinesSolid(f"{name}_lines",
                                np.vstack((heads,tails)),
                                indicies.T, 
                                line_width,
                                color = line_color)

        if self.use_weight_colors:
            tail_color = weights2rgb(np.ones(len(poses))/len(poses))
        self.tails = PcdSolid(f"{name}_tails",
                                pcd = tails,
                                color = tail_color)

    def update(self, poses : np.ndarray, weights : np.ndarray = None) -> None:
        ds = np.array([self.scale,0,0,0])
        heads = compose_s_array(poses,ds)[:,:3]
        tails = poses[:,:3]
        indicies =  np.vstack((np.arange(0,len(poses), dtype = int),
                                np.arange(len(poses),2 * len(poses), dtype = int)))

        self.lines.update(np.vstack((heads,tails)), indicies.T)
        if weights is None:
            self.tails.update(tails)
        else:
            self.tails.update(tails, color = weights2rgb(weights))

class DynamicSolid(o3dSolid):
    base_geometry : float #o3d.cuda.pybind.geometry.TriangleMesh
    pose : np.ndarray = np.zeros(4)
    
    def __init__(self, name, geometry, material, pose = None):
        self.name = name
        self.geometry = geometry
        self.material = material
        self.base_geometry = geometry
        self.pose = pose

        if pose is not None:
            self.update_geometry(pose)

    def update_geometry(self, pose : np.ndarray) -> None:
        self.geometry = deepcopy(self.base_geometry).transform(T_from_s(pose))
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
            ifc_type = element.is_a()
                      
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
                                ifc_type = ifc_type                                
                                ))

    return solids

#----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------- ADDITIONAL FUNCTIONS ----------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

#if this proves too slow, there is also this version:
#https://stackoverflow.com/questions/49156484/fast-way-to-map-scalars-to-colors-in-python
#https://stackoverflow.com/questions/66556156/getting-n-equally-spaced-rgb-colors-for-n-numbers-from-a-colormap
# CM_MAP = cm.get_cmap('jet')
N_COLORS = 100
COLORS = cm.get_cmap('jet', N_COLORS)(np.linspace(0,1,N_COLORS))[:,:3]
EPS = 1e-16
def weights2rgb(weights):
    if np.isnan(weights).any():
        raise ValueError('NaNs are not allowed in weights')
    values = (weights - weights.min()) / max(weights.max() - weights.min(), EPS) #normalize
    values = ((N_COLORS-1) * values).astype(np.int32)
    rgb = COLORS[values]
    return rgb

    # import matplotlib.pyplot as plt
    # plt.plot(np.linspace(0,1,len(weights)), weights, 'o', color = 'black')
    # plt.plot (np.linspace(0,1,len(values)), values, 'o', color = 'red')

    # weights = polyutils.mapdomain(weights, (0.0 ,1.0) , (0.0, 100.0))
    # return CM_MAP(weights)[:,:3]

def update_existence_dependence_from_yaml(solids : list[IfcSolid], yaml_filename : str) -> dict[str,str]:
    with open(yaml_filename, "r") as stream:
        try:
            existence_dependence = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return (exc)
    solids_hash = {s.name: s for s in solids}
    for key, val in existence_dependence.items():
        solids_hash[key].existence_dependence.append(solids_hash[val].name)

def remove_constructed_solids_that_cant_exist(constructed_solids : list[IfcSolid]):
    def can_exist(solid_name : str, solids_hash : dict[str,IfcSolid]):
        if len(solids_hash[solid_name].existence_dependence) == 0:
            return True
        else: 
            output = True
            for dependence_solid_name in solids_hash[solid_name].existence_dependence:
                if dependence_solid_name not in solids_hash.keys():
                    return False
                output = output and can_exist(dependence_solid_name, solids_hash)
            return output
    
    solids_hash = {s.name: s for s in constructed_solids}
    for s in constructed_solids:
        if not (can_exist(s.name, solids_hash)):
            solids_hash.pop(s.name)
    return solids_hash.values()

def compute_variation_dependence_for_rbpf(solids : list[IfcSolid]) -> list[np.ndarray]:
    solids_names = [s.name for s in solids]
    dupicate_names = {x for x in solids_names if solids_names.count(x) > 1}
    solids_varaition_dependence = []
    for name in dupicate_names:
        solid_variations = []
        for i, s_n in enumerate(solids_names):
            if s_n == name:
                solid_variations.append(i)
        solids_varaition_dependence.append(np.array(solid_variations))
    return solids_varaition_dependence

def compute_existence_dependece_for_rbpf(solids : list[IfcSolid]) -> dict[int,int]:
    solids_existence_dependence = {}
    for i, s_i in enumerate(solids):
        if s_i.existence_dependence is False: continue
        for j, s_j in enumerate(solids):
            if s_j.name in s_i.existence_dependence:
                if i in solids_existence_dependence.keys():
                    solids_existence_dependence[i] += [j]
                else: 
                    solids_existence_dependence[i] = [j]
    return solids_existence_dependence
