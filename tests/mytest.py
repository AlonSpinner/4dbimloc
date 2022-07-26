from bim4loc.binaries.paths import IFC_SIMPLE_PATH, DRONE_PATH, IFC_TEST_PATH
import open3d as o3d
import open3d.visualization.rendering as rendering
from bim4loc.geometry import pose2
from bim4loc.ifc import converter, ifcObject
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

    def add_geo(self,o : ifcObject):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLitTransparency"
        mat.base_color = np.array([1,0,0,0.5])
        o.geometry.compute_triangle_normals()
        self.vis.add_geometry(o.guid, o.geometry, mat)

objects = converter(IFC_TEST_PATH)
app = visApp()
threading.Thread(target = app.run).start()
time.sleep(0.5)

for o in objects:
    app.add_geo(o)
    # threading.Thread(target = app.add_geo, args = (o, )).start()
    time.sleep(1)