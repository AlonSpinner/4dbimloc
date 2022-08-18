
import open3d.visualization as visualization
import open3d.visualization.gui as gui
from bim4loc.solids import o3dSolid
import threading
import logging
import numpy as np
from functools import partial

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
        threading.Thread(target = self.run, name = 'app_thread').start() #executes the run method in a different thread
        
        #wait for scene and window to be created. 
        #since no window, cant use _app_thread_finished()
        while len(self._scenes) == 0:
            pass 

    def run(self) -> None:
        self.add_window("world")
        self.add_scene("world","world")   
        self._app.run()

    def add_window(self, window_name : str) -> None:
        if window_name in self._windows.keys():
            msg = "scene name already exists in VisApp's scenes"
            logging.error(msg)
            raise NameError(msg)

        def _add_window(self, window_name):
            width = 768
            height = 2 * width
            if len(self._windows) == 0:
                window = self._app.create_window(window_name, height, width)
            else:
                last_window = list(self._windows.values())[-1]
                xnew = last_window.os_frame.x + 100
                ynew = last_window.os_frame.y + 100
                window = self._app.create_window(window_name, height, width, xnew, ynew)

            self._windows[window_name] = window 

        if threading.current_thread().name == 'app_thread':
            _add_window(self, window_name)
        else:
            world_window = self._windows["world"] #we use world_window to post to main thread
            self._app.post_to_main_thread(world_window, partial(_add_window, self, window_name))
            self._app_thread_finished(world_window)

    def add_scene(self, scene_name : str, window_name : str) -> None:
        if scene_name in self._scenes.keys():
            msg = "scene name already exists in VisApp's scenes"
            logging.error(msg)
            raise NameError(msg)

        window = self._windows[window_name]        
        def _add_scene(self :'VisApp', scene_name : str, window) -> None:
            scene_widget = gui.SceneWidget()
            scene_widget.scene = visualization.rendering.Open3DScene(window.renderer)
            scene_widget.scene.set_background([1, 1, 1, 1])  # White background
            scene_widget.scene.show_ground_plane(True,visualization.rendering.Scene.GroundPlane.XY)
            scene_widget.enable_scene_caching(False)      
            window.add_child(scene_widget)
            
            self._scene2window[scene_name] = window_name
            self._scenes[scene_name] = scene_widget

            #in case we want to split the window into two scenes:
            window_scenes_names = [s for s in self._scenes.keys() if self._scene2window[s] == window_name]
            N_scenes = len(window_scenes_names)
            if  N_scenes == 3:
                msg = "window can't have more than 2 scenes"
                logging.error(msg)
                raise NameError(msg)
            elif N_scenes == 2:
                def on_layout(theme):
                    r = window.content_rect
                    self._scenes[window_scenes_names[0]].frame = gui.Rect(r.x, r.y, r.width / 2, r.height)
                    self._scenes[window_scenes_names[1]].frame= gui.Rect(r.x + r.width / 2, r.y, r.width / 2, r.height)
                window.set_on_layout(on_layout)


        if threading.current_thread().name == 'app_thread':
            _add_scene(self, scene_name, window)
        else:
            self._app.post_to_main_thread(window, partial(_add_scene,self,scene_name, window))
            self._app_thread_finished(window)

    def setup_default_camera(self, scene_name : str = "world") -> None:
        scene_widget = self._scenes[scene_name]
        
        bbox = scene_widget.scene.bounding_box
        camera = scene_widget.scene.camera

        center = bbox.get_center()
        eye = bbox.get_center() + [0,0,15]
        up = [0,1,0]
        camera.look_at(center, eye, up)

        scene_widget.center_of_rotation = center
        scene_widget.set_view_controls(scene_widget.ROTATE_CAMERA_SPHERE)

        fov = 90 #degrees
        height = scene_widget.frame.height; width = scene_widget.frame.width
        aspect_ratio = width/height
        near_plane= 0.01
        far_plane = 10 * max(bbox.get_max_bound())
        vertical = camera.FovType(0)
        camera.set_projection(fov, aspect_ratio, near_plane, far_plane, vertical)

    def add_solid(self, solid : o3dSolid, scene_name = "world") -> None:
        def _add_solid(scene_widget, solid : o3dSolid) -> None:
            scene_widget.scene.add_geometry(solid.name, solid.geometry, solid.material)
            
        scene_widget = self._scenes[scene_name]
        window = self._get_window(scene_name)
        self._app.post_to_main_thread(window, partial(_add_solid,scene_widget, solid))

    def update_solid(self, solid : o3dSolid, scene_name = "world") -> None:
        scene_widget = self._scenes[scene_name]
        if not scene_widget.scene.has_geometry(solid.name):
            logging.warning(f'geometry "{solid.name}" does not exist in scene')
            return

        def _update_solid(scene_widget, solid: o3dSolid) -> None:
            scene_widget.scene.remove_geometry(solid.name)
            scene_widget.scene.add_geometry(solid.name, solid.geometry, solid.material)

            #only works for points clouds:
            # scene_widget.scene.scene.update_geometry(solid.name, solid.geometry, solid.material)
        
        window = self._get_window(scene_name)
        self._app.post_to_main_thread(window, partial(_update_solid,scene_widget, solid))

    def set_solid_transform(self, solid: o3dSolid, T : np.ndarray, scene_name = "world") -> None:
        def _set_solid_transform(scene_widget, solid : o3dSolid, T) -> None:
            scene_widget.scene.set_geometry_transform(solid.name, T)

        scene_widget = self._scenes[scene_name]
        window = self._get_window(scene_name)
        self._app.post_to_main_thread(window, partial(_set_solid_transform,scene_widget, solid, T))

    def show_axes(self, show : bool = True, scene_name = "world") -> None:
        scene_widget = self._scenes[scene_name]
        scene_widget.scene.show_axes(show) #axes size are proportional to the existing scene size

    def redraw(self, scene_name = "world") -> None:        
        scene_widget = self._scenes[scene_name]
        window = self._get_window(scene_name)
        self._app.post_to_main_thread(window, scene_widget.force_redraw)
        self._app_thread_finished(window)

    def redraw_all_scenes(self) -> None:
        for scene_name in self._scenes.keys():
            self.redraw(scene_name)

    def _app_thread_finished(self, window) -> None:
        #blocks until app thread is finished processing all posted functions
        done_flag = [False] #send something mutable to the app thread
        def _finished(done_flag):
            done_flag[0] = True
        self._app.post_to_main_thread(window, partial(_finished, done_flag))
        while done_flag[0] == False:
            pass

    def _get_window(self, scene_name : str):
            window_name = self._scene2window[scene_name]
            window = self._windows[window_name]
            return window

if __name__ == "__main__":
    visApp = VisApp()
    print('non blocking application viewer')