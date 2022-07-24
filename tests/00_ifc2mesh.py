import ifcopenshell, ifcopenshell.geom
import numpy as np
from bim4loc.ifcs.paths import IFC_SIMPLE_PATH
import open3d as o3d

ifc = ifcopenshell.open(IFC_SIMPLE_PATH)



products = ifc.by_type("IfcProduct")
settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS,True)
settings.set(settings.APPLY_DEFAULT_MATERIALS, True)



meshes = []
for product in products:
    if product.is_a("IfcOpeningElement"): continue
    if product.Representation:
        #collect color
        shape = ifcopenshell.geom.create_shape(settings, inst=product)
        m = shape.geometry.materials
        color = np.array(m[0].diffuse)
        element = ifc.by_guid(shape.guid)
        verts = shape.geometry.verts # X Y Z of vertices in flattened list e.g. [v1x, v1y, v1z, v2x, v2y, v2z, ...]
        verts =  o3d.utility.Vector3dVector(np.array(verts).reshape((-1,3)))
        faces = shape.geometry.faces  #Indices of vertices per triangle face e.g. [f1v1, f1v2, f1v3, f2v1, f2v2, f2v3, ...]
        faces = o3d.utility.Vector3iVector(np.array(faces).reshape((-1,3)))
        mesh = o3d.geometry.TriangleMesh(vertices  = verts, triangles = faces)
        mesh.paint_uniform_color(color)

        meshes.append(mesh)

o3d.visualization.draw_geometries(meshes, mesh_show_wireframe = True)



