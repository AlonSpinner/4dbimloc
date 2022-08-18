import open3d as o3d
from open3d.visualization import rendering, gui
import random

NUM_LINES = 10

def random_point():
    return [5 * random.random(), 5 * random.random(), 5 * random.random()]

if __name__ == "__main__":
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("window", 1920, 1080)
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([1, 1, 1, 1])  # White background
    scene_widget.scene.show_ground_plane(True,rendering.Scene.GroundPlane.XY)
    scene_widget.enable_scene_caching(False)      
    window.add_child(scene_widget)

    pts = [random_point() for _ in range(0, 2 * NUM_LINES)]
    line_indices = [[2 * i, 2 * i + 1] for i in range(0, NUM_LINES)]
    colors = [[0.0, 0.0, 0.0] for _ in range(0, NUM_LINES)]

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(pts)
    lines.lines = o3d.utility.Vector2iVector(line_indices)
    lines.colors = o3d.utility.Vector3dVector(colors)

    material = rendering.MaterialRecord()
    material.shader = "unlitLine"
    material.line_width = 10 

    # add line to scene
    scene_widget.scene.add_geometry("line", lines,material)
    gui.Application.instance.run()