import open3d.visualization as visualization
import open3d.visualization.gui as gui
from bim4loc.solid_objects import o3dObject
import threading
import time

class VisApp(threading.Thread):

    def __init__(self):
        super(VisApp,self).__init__()
        
        self.start()

        #without this, we will have a bug where self._vis wont be created
        #reducing time also causes the bug
        time.sleep(0.5)

    def run(self):
        self._app = gui.Application.instance
        self._app.initialize()

        self._vis = visualization.O3DVisualizer()

        self._app.add_window(self._vis)
        self._app.run()

    def add_object(self, object : o3dObject):
        self._vis.add_geometry(object.name, object.geometry, object.material)
        
    def reset_camera_to_default(self):
        self._vis.reset_camera_to_default()
    
    def update_object(self, object : o3dObject):
        s = time.time()
        self._vis.remove_geometry(object.name)
        self._vis.add_geometry(object.name, object.geometry, object.material)
        self._vis.post_redraw()
        e = time.time()
        print(e - s)

if __name__ == "__main__":
    visApp = VisApp()
    print('non blocking application viewer')