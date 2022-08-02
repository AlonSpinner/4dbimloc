import open3d.visualization as visualization
import open3d.visualization.gui as gui
from bim4loc.solid_objects import o3dObject
import threading
import time
from typing import Literal

class VisApp():

    def __init__(self) -> None:
        threading.Thread(target = self.run).start() #executes the run method in a different thread
        
        #without this, we will have a bug where self._vis wont be created
        #reducing time also causes the bug
        time.sleep(0.5)

    def run(self) -> None:
        self._app = gui.Application.instance
        self._app.initialize()

        self._vis = visualization.O3DVisualizer()
        self.show_ground_plane(True)
        self.show_skybox(False)
        
        self._app.add_window(self._vis)
        self._app.run()

    def add_object(self, object : o3dObject) -> None:

        def _add_object(vis, object : o3dObject) -> None:
            vis.add_geometry(object.name, object.geometry, object.material)
            vis.post_redraw()

        self._app.post_to_main_thread(self._vis, lambda: _add_object(self._vis, object))
        time.sleep(0.001)

    def update_object(self, object : o3dObject) -> None:
        if not self._vis.scene.has_geometry(object.name):
            print(f'geometry {object.name} does not exist in scene')
            return

        def _update_object(vis, object: o3dObject) -> None:
            self._vis.remove_geometry(object.name)
            self._vis.add_geometry(object.name, object.geometry, object.material)
            self._vis.post_redraw()
        
        self._app.post_to_main_thread(self._vis, lambda: _update_object(self._vis, object))
        time.sleep(0.001)
    
    def redraw(self):
        self._app.post_to_main_thread(self._vis, self._vis.post_redraw)
        time.sleep(0.001)

    def reset_camera_to_default(self):
        self._app.post_to_main_thread(self._vis, self._vis.reset_camera_to_default)
        time.sleep(0.001)

    def show_axes(self, show : bool = True) -> None:
        self._vis.show_axes = show
        self.redraw()

    def show_skybox(self, show : bool = True) -> None:
        self._vis.show_skybox(show)
        self.redraw()

    def show_ground_plane(self, show : bool, ground_plane : Literal['XY','XZ','YZ']  = 'XY') -> None:
        if ground_plane == 'XY':
            self._vis.ground_plane = visualization.rendering.Scene.GroundPlane(1)
        elif ground_plane == 'XZ':
            self._vis.ground_plane = visualization.rendering.Scene.GroundPlane(0)
        elif ground_plane == 'YZ':
            self._vis.ground_plane = visualization.rendering.Scene.GroundPlane(2)
        
        if show:
            self._vis.show_ground = True  
        
        self.redraw()

if __name__ == "__main__":
    visApp = VisApp()
    print('non blocking application viewer')