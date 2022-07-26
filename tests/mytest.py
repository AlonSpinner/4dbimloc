from bim4loc.binaries.paths import IFC_SIMPLE_PATH, DRONE_PATH, IFC_TEST_PATH
import open3d as o3d
import open3d.visualization.rendering as rendering
from bim4loc.geometry import pose2
from bim4loc.ifc import converter
import time
import numpy as np
import open3d as o3d
import threading
import time


class visApp():

    def __init__(self):
        self.is_done = False
    
    def run(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.vis = o3d.visualization.O3DVisualizer('vis')
        self.vis.set_on_close(self.on_main_window_closing)

        app.add_window(self.vis)
        
        app.run()


    def on_main_window_closing(self):
        self.is_done = True
        return True  # False would cancel the close

    def add_geo(self,geo):
        self.vis.add_geometry(geo)


    

objects = converter(IFC_TEST_PATH)
app = visApp()
threading.Thread(target = app.run()).start()

for o in objects:
    threading.Thread(target = app.add_geo(o.geometry)).start()
    time.sleep(1)