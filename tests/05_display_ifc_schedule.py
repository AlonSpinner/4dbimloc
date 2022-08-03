from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, IFC_TEST_PATH
from bim4loc.visualizer2 import VisApp
from bim4loc.solids import ifc_converter
from bim4loc.random_models.one_dim import Gaussian
import open3d.visualization.gui as gui
from threading import Thread

# import open3d as o3d
# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

app = gui.Application.instance
app.initialize()
Thread(target = app.run).start()


solids = ifc_converter(IFC_ONLY_WALLS_PATH)
visApp1 = VisApp()
current_time = 5.0

for s in solids:
    if isinstance(s.schedule,Gaussian): #should be all of 'em
        s.set_random_completion_time()
        if s.is_complete(current_time):
        # s.set_shader_by_schedule_and_time(current_time)
            visApp1.add_solid(s)

#show axes and reset camera after building scene
visApp1.show_axes(True)
visApp1.reset_camera_to_default()



