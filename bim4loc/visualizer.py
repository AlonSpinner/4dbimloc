import open3d.visualization as visualization
import open3d.visualization.gui as gui
from bim4loc.ifc import ifcObject
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

    def add_object(self, object : ifcObject):
        self._vis.add_geometry(object.name, object.geometry, object.material)

if __name__ == "__main__":
    visApp = VisApp()
    print('non blocking application viewer')