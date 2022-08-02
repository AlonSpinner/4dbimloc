from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, IFC_TEST_PATH
from bim4loc.solids import ifc_converter
import trimesh
import numpy as np
import pyrender 

scene = pyrender.Scene()
cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)

solids = ifc_converter(IFC_ONLY_WALLS_PATH)

for s in solids:
    tm = trimesh.base.Trimesh(vertices = s.geometry.vertices, faces = s.geometry.triangles)
    prm = pyrender.Mesh.from_trimesh(tm)
    scene.add(prm)

viewer = pyrender.Viewer(scene, use_direct_lighting=True,
                run_in_thread=True, 
                central_node = prm,
                show_world_axis = True)


