import open3d.visualization as visualization
import open3d.visualization.gui as gui
from bim4loc.solids import o3dSolid
import threading
import time
from typing import Literal
import logging

class VisApp():

    def __init__(self) -> None:
        self._windows = [] #list of windows

        self._lock = threading.Lock()
        threading.Thread(target = self.run).start() #executes the run method in a different thread
        
        #without this, we will have a bug where self._vis wont be created
        #reducing time also causes the bug
        time.sleep(0.5)

    def run(self) -> None:
        self._app = gui.Application.instance
        self._app.initialize()
        
        self.add_window() #must create app with at least one window or core dumps
        self._app.run()

    def set_active_window(self, n : int):
        self._active_window = self._windows[n]

    def add_window(self, title : str = ''):
        def _add_window(self):
            new_window = visualization.O3DVisualizer(title)
            self._app.add_window(new_window)

            self._windows.append(new_window)
            self._active_window = new_window
            
            self.show_ground_plane(True)
            self.show_skybox(False)

        if len(self._windows) == 0:
            _add_window(self)
        else:
            self._lock.acquire()
            self._app.post_to_main_thread(self._active_window, lambda: _add_window(self)) #this works god knows why
            self._lock.release()

    def add_solid(self, solid : o3dSolid) -> None:
        self._lock.acquire()

        def _add_solid(window, solid : o3dSolid) -> None:
            window.add_geometry(solid.name, solid.geometry, solid.material)
            window.post_redraw()

        self._app.post_to_main_thread(self._active_window, lambda: _add_solid(self._active_window, solid))
        self._lock.release()

    def update_solid(self, solid : o3dSolid) -> None:
        self._lock.acquire()

        if not self._active_window.scene.has_geometry(solid.name):
            logging.warning(f'geometry {solid.name} does not exist in scene')
            return

        def _update_solid(window, solid: o3dSolid) -> None:
            window.remove_geometry(solid.name)
            window.add_geometry(solid.name, solid.geometry, solid.material)
            window.post_redraw()
        
        self._app.post_to_main_thread(self._active_window, lambda: _update_solid(self._active_window, solid))
        self._lock.release()
    
    def redraw(self):
        self._lock.acquire()
        self._app.post_to_main_thread(self._active_window, self._active_window.post_redraw)
        self._lock.release()

    def reset_camera_to_default(self):
        self._lock.acquire()
        self._app.post_to_main_thread(self._active_window, self._active_window.reset_camera_to_default)
        self._lock.release()

    def show_axes(self, show : bool = True) -> None:
        self._active_window.show_axes = show #axes size are proportional to the scene size
        self.redraw()

    def show_skybox(self, show : bool = True) -> None:
        self._active_window.show_skybox(show)
        self.redraw()

    def show_ground_plane(self, show : bool, ground_plane : Literal['XY','XZ','YZ']  = 'XY') -> None:
        if ground_plane == 'XY':
            self._active_window.ground_plane = visualization.rendering.Scene.GroundPlane(1)
        elif ground_plane == 'XZ':
            self._active_window.ground_plane = visualization.rendering.Scene.GroundPlane(0)
        elif ground_plane == 'YZ':
            self._active_window.ground_plane = visualization.rendering.Scene.GroundPlane(2)
        
        if show:
            self._active_window.show_ground = True  
        
        self.redraw()

if __name__ == "__main__":
    visApp = VisApp()
    # time.sleep(1)
    
    visApp.add_window()
    print('non blocking application viewer')