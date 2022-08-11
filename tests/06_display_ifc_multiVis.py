from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, IFC_TEST_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter
import logging
import time
import threading

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

solids = ifc_converter(IFC_ONLY_WALLS_PATH)
visApp = VisApp()

for s in solids:
    visApp.add_solid(s)

visApp.show_axes(True)
visApp.reset_camera_to_default()

visApp.add_window('second_window')
for s in solids:
    visApp.add_solid(s)
visApp.show_axes(True)
visApp.reset_camera_to_default()


