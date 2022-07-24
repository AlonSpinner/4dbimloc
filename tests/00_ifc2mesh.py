from bim4loc.binaries.paths import IFC_SIMPLE_PATH, DRONE_PATH
import open3d as o3d
from bim4loc.geometry import pose2
from bim4loc.ifc2o3d import converter

meshes = converter(IFC_SIMPLE_PATH)

frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

drone = o3d.io.read_triangle_mesh(DRONE_PATH)
drone.paint_uniform_color([1,0,0])
drone.transform(pose2(3,3,0).T3d(z = 1.5))

o3d.visualization.draw_geometries(meshes + [drone] + [frame], mesh_show_wireframe = True)