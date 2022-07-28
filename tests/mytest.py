from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, DRONE_PATH
import open3d as o3d
import open3d.visualization.rendering as rendering
from bim4loc.geometry import pose2
from bim4loc.solid_objects import converter, ifcObject
import time
import numpy as np
import open3d as o3d
import threading
import time


class visApp(threading.Thread):

    def __init__(self):
        super(visApp,self).__init__()

        self.is_done = False
        self.start()
    
    def run(self):
        self.app = o3d.visualization.gui.Application.instance
        self.app.initialize()

        self.vis = o3d.visualization.O3DVisualizer('vis')
        self.vis.set_on_close(self.on_main_window_closing)

        self.app.add_window(self.vis)   
        self.app.run()


    def on_main_window_closing(self):
        self.is_done = True
        return True  # False would cancel the close

    def add_geo(self,o : ifcObject):
        # mat.shader = "defaultLitTransparency"
        self.vis.add_geometry(o.name, o.geometry, o.material)

objects = converter(IFC_ONLY_WALLS_PATH)
app = visApp()
# threading.Thread(target = app.run).start()
time.sleep(0.5)

for o in objects:
    app.add_geo(o)
    # threading.Thread(target = app.add_geo, args = (o, )).start()
    time.sleep(1)