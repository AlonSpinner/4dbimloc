from bim4loc.binaries.paths import IFC_SIMPLE_PATH, DRONE_PATH, IFC_TEST_PATH
import open3d as o3d
import open3d.visualization.rendering as rendering
from bim4loc.geometry import pose2
from bim4loc.ifc import converter
import time

objects = converter(IFC_TEST_PATH)
t = 4.0 #current time

frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

drone = o3d.io.read_triangle_mesh(DRONE_PATH)
drone.paint_uniform_color([1,0,0])
drone.transform(pose2(3,3,0).T3d(z = 1.5))

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().mesh_show_wireframe = True

#TRYING TO DO ORTHOGRAPHIC CAMERA
# K = vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic
# width = int(K.intrinsic_matrix[0,2] * 2)
# height = int(K.intrinsic_matrix[1,2] * 2)
# fx, fy = 1000000000.0, 1000000000.0
# cx, cy = 0.0, 0.0
# K.set_intrinsics(width,height,fx,fy,cx,cy)

for o in objects:
    mat = rendering.MaterialRecord()
    # mat.base_coor = o.color
    # mat.shader = "defaultLit"
    # o.opacity(time)
    vis.add_geometry(o.geometry)

vis.poll_events()
vis.update_renderer()
vis.run()
time.sleep(50)
vis.close_window()