import numpy as np
from bim4loc.binaries.paths import IFC_SIMPLE_PATH, DRONE_PATH
import open3d as o3d
from bim4loc.geometry import pose2
from bim4loc.ifc import converter
import time
import asyncio

async def show_scene(vis, geometries):
        print('no')
        vis.create_window()
        vis.get_render_option().mesh_show_wireframe = True
        for g in geometries:
            vis.add_geometry(g)
        vis.run()


async def main():
    objects = converter(IFC_SIMPLE_PATH)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    drone = o3d.io.read_triangle_mesh(DRONE_PATH)
    drone.paint_uniform_color([1,0,0])
    x = pose2(3,3,0)
    drone.transform(x.T3d(z = 1.5))

    scene_geometries = [(o.geometry) for o in objects] + [frame] + [drone]

    vis = o3d.visualization.Visualizer()
    task_show =  asyncio.create_task(show_scene(vis, scene_geometries))
    
    actions = [pose2(1,0,np.pi/10)] * 10


    # vis.run() # allows to set viewpoint and interact with the scene
    for a in actions:
        drone.rotate(a.R3d(),x.t3d())
        drone.translate(a.t3d())

        vis.update_geometry(drone)
        vis.poll_events()
        vis.update_renderer()

        x = x + a
        print('yes')
        time.sleep(0.1)

    time.sleep(50.0)
    vis.destroy_window()

asyncio.run(main())