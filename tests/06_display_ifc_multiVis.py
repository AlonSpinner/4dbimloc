from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, IFC_TEST_PATH
from bim4loc.visualizer2 import VisApp
from bim4loc.solids import ifc_converter

# import open3d as o3d
# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

solids = ifc_converter(IFC_ONLY_WALLS_PATH)
visApp = VisApp()

for s in solids:
    visApp.add_solid(s)

#show axes and reset camera after building scene
visApp.show_axes(True)
visApp.reset_camera_to_default()

visApp.add_window()
for s in solids:
    visApp.add_solid(s)
visApp.show_axes(True)
visApp.reset_camera_to_default()

