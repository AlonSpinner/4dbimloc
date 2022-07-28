from dataclasses import dataclass
import ifcopenshell, ifcopenshell.geom
import numpy as np
import open3d as o3d
import bim4loc.random_models.one_dim as random1d


@dataclass(frozen = False)
class ifcObject:
    name : str #name
    geometry : o3d.cuda.pybind.geometry.TriangleMesh
    material : o3d.cuda.pybind.visualization.rendering.MaterialRecord
    schedule : random1d.distribution1D
    completion_time : float = 0.0
    
    def random_completion_time(self) -> None:
        self.completion_time = self.schedule.sample()
    def opacity(self, time: float) -> float:
        return self.schedule.cdf(time)
    def complete(self, time : float) -> bool:
        return time > self.completion_time

def description2schedule(description : str) -> random1d.distribution1D:
    if description:
        lst = description.split(" ")
        if lst[0] == "gaussian":
            return random1d.gaussian(mu = float(lst[1]), sigma = float(lst[2]))
        
        elif lst[0] == "uniform":
            return random1d.uniform(a = float(lst[1]), b = float(lst[2]))

        elif lst[0] == "gaussianT":
            return random1d.gaussianT(mu = lst[1], sigma = float(lst[2]), a = float(lst[3]), b = float(lst[4]))
        
    else:
        return random1d.distribution1D() #empty

def converter(ifc_path) -> list[ifcObject]:
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

            objects.append(ifcObject(
                                name = element.GlobalId,
                                geometry = mesh,
                                material = mat,
                                schedule = description2schedule(element.Description),                                
                                ))

    return objects
