from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, IFC_TEST_PATH
# from bim4loc.visualizer import VisApp
from bim4loc.solid_objects import ifc_converter
import time
import trimesh
import numpy as np
import pyrender 

# import open3d as o3d
# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

scene = pyrender.Scene()
cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
# cam_node = scene.add(cam, pose=cam_pose)

objects = ifc_converter(IFC_ONLY_WALLS_PATH)

for o in objects:
    tm = trimesh.base.Trimesh(vertices = o.geometry.vertices, faces = o.geometry.triangles)
    prm = pyrender.Mesh.from_trimesh(tm)
    scene.add(prm)

viewer = pyrender.Viewer(scene, use_direct_lighting=True,
                run_in_thread=True, 
                central_node = prm,
                show_world_axis = True)
    

# #show axes and reset camera after building scene
# visApp.show_axes(True)
# visApp.reset_camera_to_default()


