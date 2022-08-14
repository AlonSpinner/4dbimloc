#from https://github.com/isl-org/Open3D/issues/999

#!/usr/bin/env python

import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np

def make_box():
    box = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    box.compute_vertex_normals()
    return box

def main():
    gui.Application.instance.initialize()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.base_color = (1.0, 0.0, 0.0, 1.0)
    mat.shader = "defaultLit"

    w = gui.Application.instance.create_window("Two scenes", 1025, 512)
    scene1 = gui.SceneWidget()
    scene1.scene = o3d.visualization.rendering.Open3DScene(w.renderer)
    scene1.scene.add_geometry("cube1", make_box(), mat)
    scene1.setup_camera(60, scene1.scene.bounding_box, (0, 0, 0))
    scene2 = gui.SceneWidget()
    scene2.scene = o3d.visualization.rendering.Open3DScene(w.renderer)
    mat.base_color = (0.0, 0.0, 1.0, 1.0)
    scene2.scene.add_geometry("cube1", make_box(), mat)
    scene2.setup_camera(60, scene2.scene.bounding_box, (0, 0, 0))

    w.add_child(scene1)
    w.add_child(scene2)

    def on_layout(theme):
        r = w.content_rect
        scene1.frame = gui.Rect(r.x, r.y, r.width / 2, r.height)
        scene2.frame = gui.Rect(r.x + r.width / 2 + 1, r.y, r.width / 2, r.height)

    w.set_on_layout(on_layout)

    gui.Application.instance.run()

if __name__ == "__main__":
    main()