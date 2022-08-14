from curses import keyname
from distutils.core import setup
from tkinter import CENTER
import open3d.visualization as visualization
import open3d.visualization.gui as gui
from bim4loc.solids import o3dSolid
import threading
import time
from typing import Literal
import logging
import numpy as np

# gui.Application : http://www.open3d.org/docs/release/python_api/open3d.visualization.gui.Application.html#open3d.visualization.gui.Application
# gui.Window :  http://www.open3d.org/docs/release/python_api/open3d.visualization.gui.Window.html?highlight=gui%20application%20instance%20create_window
# sceneWidget : http://www.open3d.org/docs/release/python_api/open3d.visualization.gui.SceneWidget.html?highlight=scenewidget#open3d.visualization.gui.SceneWidget.scene
#rendering.Open3DScene : http://www.open3d.org/docs/release/python_api/open3d.visualization.rendering.Open3DScene.html?highlight=rendering%20open3dscene#open3d.visualization.rendering.Open3DScene
# camera : http://www.open3d.org/docs/release/python_api/open3d.visualization.rendering.Camera.html

class VisApp():

    def __init__(self) -> None:
        self._scenes : dict[gui.SceneWidget] = {}
        self._windows : dict = {}
        self._scene2window : dict[str] = {}
        self._scene_heightWidth : dict[tuple] = {}

        self._app = gui.Application.instance
        self._app.initialize()
        threading.Thread(target = self.run).start() #executes the run method in a different thread
        time.sleep(0.1) #some time for self.add_scene to finish

    def run(self) -> None:
        self.add_scene("world")   
        self._app.run()

    def add_scene(self, scene_name : str, window_name : str = None):
        if scene_name in self._scenes.keys():
            msg = "scene name already exists in VisApp's scenes"
            logging.error(msg)
            raise NameError(msg)

        if window_name not in self._windows.keys():
            #create new window having the same name as scene
            window_name = scene_name
            width = 768
            height = 2 * width
            window = self._app.create_window(window_name, height, width)
            self._windows[window_name] = window #add to _windows
        else:
            window = self._windows[window_name] = window

        scene_widget = gui.SceneWidget()
        scene_widget.scene = visualization.rendering.Open3DScene(window.renderer)
        scene_widget.scene.set_background([1, 1, 1, 1])  # White background
        scene_widget.scene.show_ground_plane(True,visualization.rendering.Scene.GroundPlane.XY)
        scene_widget.enable_scene_caching(False)

        window.add_child(scene_widget)
        self._scenes[scene_name] = scene_widget
        self._scene2window[scene_name] = window_name
        self._scene_heightWidth[scene_name] = (height,width)

    def setup_default_camera(self, scene_name : str = "world") -> None:
        scene_widget = self._scenes[scene_name]
        height,width = self._scene_heightWidth[scene_name]
        
        bbox = scene_widget.scene.bounding_box
        camera = scene_widget.scene.camera

        center = bbox.get_center()
        eye = bbox.get_center() + [0,0,15]
        up = [0,1,0]
        camera.look_at(center, eye, up)

        scene_widget.center_of_rotation = center
        scene_widget.set_view_controls(scene_widget.ROTATE_CAMERA_SPHERE)

        fov = 90 #degrees
        aspect_ratio = height/width
        near_plane= 0.1
        far_plane = 10 * max(bbox.get_max_bound())
        vertical = camera.FovType(1)
        camera.set_projection(fov, aspect_ratio, near_plane, far_plane, vertical)

    def add_solid(self, solid : o3dSolid, scene_name = 'world') -> None:
        def _add_solid(scene_widget, solid : o3dSolid) -> None:
            scene_widget.scene.add_geometry(solid.name, solid.geometry, solid.material)

        scene_widget = self._scenes[scene_name]
        window = self._get_window(scene_name)
        self._app.post_to_main_thread(window, lambda: _add_solid(scene_widget, solid))

    def set_solid_transform(self, solid: o3dSolid, T : np.ndarray, scene_name = 'world') -> None:
        def _set_solid_transform(scene_widget, solid : o3dSolid, T) -> None:
            scene_widget.scene.set_geometry_transform(solid.name, T)

        scene_widget = self._scenes[scene_name]
        window = self._get_window(scene_name)
        self._app.post_to_main_thread(window, lambda: _set_solid_transform(scene_widget, solid, T))

    def update_solid(self, solid : o3dSolid, scene_name = 'world') -> None:
        scene_widget = self._scenes[scene_name]
        if not scene_widget.scene.has_geometry(solid.name):
            logging.warning(f'geometry {solid.name} does not exist in scene')
            return

        def _update_solid(scene_widget, solid: o3dSolid) -> None:
            scene_widget.scene.remove_geometry(solid.name)
            scene_widget.scene.add_geometry(solid.name, solid.geometry, solid.material)
        
        window = self._get_window(scene_name)
        self._app.post_to_main_thread(window, lambda: _update_solid(scene_widget, solid))

    def show_axes(self, scene_name = 'world' ,show : bool = True) -> None:
        scene_widget = self._scenes[scene_name]
        #Important: Axes need to be drawn AFTER the app has finished adding all relevent solids
        scene_widget.scene.show_axes(show) #axes size are proportional to the scene size
        
    def redraw(self, scene_name = 'world'):
        sceneWidget = self._scenes[scene_name]
        window = self._get_window(scene_name)
        self._app.post_to_main_thread(window, sceneWidget.force_redraw)

    def _get_window(self, scene_name : str):
            window_name = self._scene2window[scene_name]
            window = self._windows[window_name]
            return window

if __name__ == "__main__":
    visApp = VisApp()
    print('non blocking application viewer')