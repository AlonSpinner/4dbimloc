import ifcopenshell, ifcopenshell.geom
import numpy as np
from bim4loc.binaries.paths import IFC_SIMPLE_PATH, DRONE_PATH
import open3d as o3d
from bim4loc.geometry import pose2
from bim4loc.ifc import converter
import time

#based on http://www.open3d.org/docs/release/tutorial/visualization/non_blocking_visualization.html

objects = converter(IFC_SIMPLE_PATH)
frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
drone = o3d.io.read_triangle_mesh(DRONE_PATH)
drone.paint_uniform_color([1,0,0])
x = pose2(3,3,0)
drone.transform(x.T3d(z = 1.5))

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
vis = o3d.visualization.Visualizer()

vis.create_window()
[vis.add_geometry(o.geometry) for o in objects]
vis.add_geometry(frame)
vis.add_geometry(drone)

actions = [pose2(1,0,np.pi/10)] * 10

# vis.run() # allows to set viewpoint and interact with the scene
for a in actions:
    drone.rotate(a.R3d(),x.t3d())
    drone.translate(a.t3d())

    # vis.update_geometry(drone)
    vis.poll_events()
    vis.update_renderer()

    x = x + a
    time.sleep(0.1)

time.sleep(50.0)
vis.destroy_window()
