from asyncio import base_events
from dataclasses import dataclass
import ifcopenshell, ifcopenshell.geom
import numpy as np
import open3d as o3d
import bim4loc.random_models.one_dim as random1d
from bim4loc.geometry import pose2
from importlib import import_module
from copy import deepcopy

@dataclass(frozen = False)
class o3dObject:
    name : str #name
    geometry : o3d.cuda.pybind.geometry.TriangleMesh
    material : o3d.cuda.pybind.visualization.rendering.MaterialRecord

@dataclass()
class IfcObject(o3dObject):
    schedule : random1d.Distribution1D
    completion_time : float = 0.0
    
    def random_completion_time(self) -> None:
        self.completion_time = self.schedule.sample()
    def opacity(self, time: float) -> float:
        return self.schedule.cdf(time)
    def complete(self, time : float) -> bool:
        return time > self.completion_time

class DynamicObject(o3dObject):
    base_geometry : o3d.cuda.pybind.geometry.TriangleMesh
    pose : pose2 = pose2(0,0,0)
    
    def __init__(self, name, geometry, material, pose = None):
        self.name = name
        self.geometry = geometry
        self.material = material
        self.base_geometry = geometry
        self.pose = pose

        if pose is not None:
            self.update_geometry(pose)

    def update_geometry(self, pose : pose2, z = 0) -> None:
        self.geometry = deepcopy(self.base_geometry).transform(pose.T3d(z = z))
        self.pose = pose

#----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------- IFC CONVERTION ----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------


def description2schedule(description : str) -> random1d.Distribution1D:
    if description:
        try:
            lst = description.split(" ")
            _dname = lst[0]
            _dparams = [int(num) for num in lst[1:]]
            _class = getattr(import_module(random1d.__name__),_dname)
            instance = _class.__new__(_class)
            instance.__init__(*_dparams)
            return instance
        except:
            print(f'description does not fit any programmed schedule time distribution in {random1d.__name__}')
    else:
        return random1d.Distribution1D() #empty

def ifc_converter(ifc_path) -> list[IfcObject]:
    '''
    converts ifc file to a list of ifcObjects
    '''
    ifc = ifcopenshell.open(ifc_path)

    products = ifc.by_type("IfcProduct")
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS,True)
    settings.set(settings.APPLY_DEFAULT_MATERIALS, True)

    objects = []
    for product in products:
        if product.is_a("IfcOpeningElement"): continue
        if product.Representation: #has shape
            shape = ifcopenshell.geom.create_shape(settings, inst=product)
            m = shape.geometry.materials
            base_color = np.array(m[0].diffuse)
            element = ifc.by_guid(shape.guid)
                      
            verts = shape.geometry.verts # X Y Z of vertices in flattened list e.g. [v1x, v1y, v1z, v2x, v2y, v2z, ...]
            verts =  o3d.utility.Vector3dVector(np.array(verts).reshape((-1,3)))
            
            faces = shape.geometry.faces  #Indices of vertices per triangle face e.g. [f1v1, f1v2, f1v3, f2v1, f2v2, f2v3, ...]
            faces = o3d.utility.Vector3iVector(np.array(faces).reshape((-1,3)))
            
            mesh = o3d.geometry.TriangleMesh(vertices  = verts, triangles = faces)
            mesh.compute_triangle_normals()
            
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency"
            mat.base_color = np.hstack([base_color, 1.0])

            objects.append(IfcObject(
                                name = element.GlobalId,
                                geometry = mesh,
                                material = mat,
                                schedule = description2schedule(element.Description),                                
                                ))

    return objects
