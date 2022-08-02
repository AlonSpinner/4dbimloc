from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, IFC_TEST_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solid_objects import ifc_converter
import time

import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

objects = ifc_converter(IFC_ONLY_WALLS_PATH)
visApp = VisApp()

for o in objects:
    visApp.add_object(o); time.sleep(0.001)

#show axes and reset camera after building scene
visApp.show_axes(True)
visApp.reset_camera_to_default()


