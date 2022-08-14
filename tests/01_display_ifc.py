from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, IFC_TEST_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter
import logging
import time

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

solids = ifc_converter(IFC_ONLY_WALLS_PATH)
visApp = VisApp()

for s in solids:
    visApp.add_solid(s, "world")

visApp.redraw()
time.sleep(1) #need to wait until all solids are rendered 
visApp.setup_default_camera("world")
visApp.show_axes()



