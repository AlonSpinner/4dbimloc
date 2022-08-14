from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, IFC_TEST_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter
import logging
import time

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

solids = ifc_converter(IFC_ONLY_WALLS_PATH)
visApp = VisApp()
# time.sleep(0.2)
for s in solids:
    visApp.add_solid(s)
    # time.sleep(0.1)

visApp.setup_default_camera('world')
# time.sleep(0.2)


#show axes and reset camera after building scene
# visApp.show_axes(True)
# visApp.reset_camera_to_default()


