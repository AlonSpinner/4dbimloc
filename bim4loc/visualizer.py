from distutils.core import setup
import open3d.visualization as visualization
import open3d.visualization.gui as gui
from bim4loc.solids import o3dSolid
import threading
import time
from typing import Literal
import logging

# gui.Application doc: http://www.open3d.org/docs/release/python_api/open3d.visualization.gui.Application.html#open3d.visualization.gui.Application
# sceneWidget doc : http://www.open3d.org/docs/release/python_api/open3d.visualization.gui.SceneWidget.html?highlight=scenewidget#open3d.visualization.gui.SceneWidget.scene

class VisApp():

    def __init__(self) -> None:
        self._scenes : dict[gui.SceneWidget] = {} #list of scebes in window
        self._lock = threading.Lock()

        threading.Thread(target = self.run).start() #executes the run method in a different thread
        # time.sleep(0.5) #amount of time it takes app to create the first window

    def run(self) -> None:
        with self._lock:
            self._app = gui.Application.instance
            self._app.initialize()

            window = gui.Application.instance.create_window("App Window", 1025, 512)
            worldSceneWidget = gui.SceneWidget()
            worldSceneWidget.scene = visualization.rendering.Open3DScene(window.renderer)
            worldSceneWidget.scene.set_background([1, 1, 1, 1])  # White background
            worldSceneWidget.scene.show_ground_plane(True,visualization.rendering.Scene.GroundPlane.XY)
            worldSceneWidget.enable_scene_caching(False)
            # self.setup_default_camera(worldSceneWidget)

            window.add_child(worldSceneWidget)
            
            self.window = window
            self._scenes['world'] = worldSceneWidget
        
        self._app.run()

    def setup_default_camera(self, scene_name : str) -> None:
        with self._lock:
            sceneWidget = self._scenes[scene_name]
            bbox = sceneWidget.scene.bounding_box
            sceneWidget.look_at(bbox.get_center(), bbox.get_center() + [0,0,10], [0,-1,0])
            sceneWidget.force_redraw()
        # fov = 60 #deggrees
        # setup_camera(fov, sceneWidget.bounds.get_center(),
                            #  bounds.get_center() + [0, 0, -3], [0, -1, 0])
        
        # sceneWidget.setup_camera(60, sceneWidget.scene.bounding_box, (0, 0, 10))
        # bounds = sceneWidget.scene.bounding_box
        # sceneWidget.setup_camera(60.0, bounds, bounds.get_center())
        # sceneWidget.scene.camera.look_at([0, 0, 0], [0, 0, 10], [0, 1, 0])                             
            # self._scenes[scene_name].scene.reset_view_point(True)
            pass

    def set_active_window(self, n : int):
            self._active_window = self._windows[n]

    # def add_window(self, title : str = ''):
    #     with self._lock:
    #         #for addtional windows beyond the first one
    #         def _add_window(self : 'VisApp'):
    #             new_window = visualization.O3DVisualizer(title)
    #             self._app.add_window(new_window)

    #             #has to be after window is added to app
    #             last_window_pos = (self._windows[-1].os_frame.x, self._windows[-1].os_frame.y)
    #             new_window.os_frame = gui.Rect(last_window_pos[0] + 1800,
    #                                         last_window_pos[1] + 50,
    #                                         new_window.os_frame.width,
    #                                         new_window.os_frame.height)

    #             self._windows.append(new_window)
    #             self._active_window = new_window
                
    #         self._app.post_to_main_thread(self._active_window, lambda: _add_window(self)) #this works god knows why
    #     self.show_ground_plane(True)
    #     self.show_skybox(False)

    def add_solid(self, solid : o3dSolid, sceneName = 'world') -> None:
        with self._lock:
            def _add_solid(sceneWidget, solid : o3dSolid) -> None:
                sceneWidget.scene.add_geometry(solid.name, solid.geometry, solid.material)
                # sceneWidget.scene.scene.post_redraw()

            sceneWidget = self._scenes[sceneName]
            self._app.post_to_main_thread(self.window, lambda: _add_solid(sceneWidget, solid))
            sceneWidget.force_redraw()
            # time.sleep(0.001) #amount of time it takes app to add the solid
            # t = self._app.post_to_main_thread(self._active_window, self._app.run_one_tick)

    def update_solid(self, solid : o3dSolid) -> None:
        with self._lock:
            if not self._active_window.scene.has_geometry(solid.name):
                logging.warning(f'geometry {solid.name} does not exist in scene')
                return

            def _update_solid(window, solid: o3dSolid) -> None:
                window.remove_geometry(solid.name)
                window.add_geometry(solid.name, solid.geometry, solid.material)
                window.post_redraw()
            
            self._app.post_to_main_thread(self._active_window, lambda: _update_solid(self._active_window, solid))
            # time.sleep(0.001) #amount of time it takes app to update the solid
    
    def redraw(self, blocking = True):
        with self._lock:
            self._active_window.post_redraw()

    def reset_camera_to_default(self):
        with self._lock:
            self._app.post_to_main_thread(self._active_window, self._active_window.reset_camera_to_default)

    def show_axes(self, show : bool = True) -> None:
        #Important: Axes need to be drawn AFTER the app has finished adding all relevent solids
        self._active_window.show_axes = show #axes size are proportional to the scene size
        self.redraw(blocking = True)

    def show_skybox(self, show : bool = True) -> None:
        self._active_window.show_skybox(show)
        self.redraw(blocking = False)

    def show_ground_plane(self, show : bool, ground_plane : Literal['XY','XZ','YZ']  = 'XY') -> None:
        if ground_plane == 'XY':
            self._active_window.ground_plane = visualization.rendering.Scene.GroundPlane(1)
        elif ground_plane == 'XZ':
            self._active_window.ground_plane = visualization.rendering.Scene.GroundPlane(0)
        elif ground_plane == 'YZ':
            self._active_window.ground_plane = visualization.rendering.Scene.GroundPlane(2)
        
        if show:
            self._active_window.show_ground = True  
        
        self.redraw(blocking = False)

if __name__ == "__main__":
    visApp = VisApp()
    
    visApp.add_window()
    print('non blocking application viewer')