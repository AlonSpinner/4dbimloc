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
time.sleep(0.5)  #need to wait until all solids are rendered 

for s in solids:
    visApp.add_solid(s)
visApp.redraw()
visApp.setup_default_camera("world")
visApp.show_axes()

visApp.add_window("belief")
visApp.add_scene("belief", "belief")
for s in solids:
    visApp.add_solid(s, "belief")
visApp.redraw()
visApp.setup_default_camera("belief")
visApp.show_axes("belief", True)


