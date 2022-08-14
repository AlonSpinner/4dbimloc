from curses import keyname
from distutils.core import setup
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
        self._lock = threading.Lock() #lock is mostly for first window. don't want to update stuff before fully creating

        self._app = gui.Application.instance
        self._app.initialize()
        threading.Thread(target = self.run).start() #executes the run method in a different thread
        time.sleep(0.1) 

    def run(self) -> None:
        self.add_scene("world")   
        self._app.run()

    def setup_default_camera(self, scene_name : str = "world") -> None:
        with self._lock:
            sceneWidget = self._scenes[scene_name]
            bbox = sceneWidget.scene.bounding_box
            camera = sceneWidget.scene.camera
            camera.look_at(bbox.get_center(), bbox.get_center() + [0,0,15], [0,1,0])

            fov = 90
            aspect_ratio = 2
            near_plane= 0.0
            far_plane = 10.0
            vertical = camera.FovType(0.1)
            camera.set_projection(fov, aspect_ratio, near_plane, far_plane, vertical)
            # print(camera.get_field_of_view())
            # print(camera.get_projection_matrix())

    def setup_default_camera2(self, scene_name : str = "world") -> None:
        with self._lock:
            sceneWidget = self._scenes[scene_name]
            bbox = sceneWidget.scene.bounding_box
            camera = sceneWidget.scene.camera

            sceneWidget.look_at(bbox.get_center(), bbox.get_center() + [0,0,10], [0,1,0])
            print(camera.get_field_of_view())
            print(camera.get_projection_matrix())

    def add_scene(self, scene_name : str, window_name : str = None):
        with self._lock:
            if scene_name in self._scenes.keys():
                msg = "scene name already exists in VisApp's scenes"
                logging.error(msg)
                raise NameError(msg)

            if window_name not in self._windows.keys():
                #create new window having the same name as scene
                window_name = scene_name
                window = self._app.create_window(window_name, int(1.5*1025), int(1.5*512))
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

    def add_solid(self, solid : o3dSolid, scene_name = 'world') -> None:
        with self._lock:
            def _add_solid(scene_widget, solid : o3dSolid) -> None:
                scene_widget.scene.add_geometry(solid.name, solid.geometry, solid.material)

            scene_widget = self._scenes[scene_name]
            window = self._get_window(scene_name)
            self._app.post_to_main_thread(window, lambda: _add_solid(scene_widget, solid))

    def set_solid_transform(self, solid: o3dSolid, T : np.ndarray, scene_name = 'world') -> None:
        #currently doesnt work? dont know why
        with self._lock:
            def _set_solid_transform(scene_widget, solid : o3dSolid, T) -> None:
                scene_widget.scene.set_geometry_transform(solid.name, T)

            scene_widget = self._scenes[scene_name]
            window = self._get_window(scene_name)
            self._app.post_to_main_thread(window, lambda: _set_solid_transform(scene_widget, solid, T))

    def update_solid(self, solid : o3dSolid, scene_name = 'world') -> None:
        with self._lock:
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
        with self._lock:
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