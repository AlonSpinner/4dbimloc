from bim4loc.binaries.paths import IFC_ARENA_PLUS_PATH as IFC_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter
import logging

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)


solids = ifc_converter(IFC_PATH)
visApp = VisApp()

for i, s in enumerate(solids):
    visApp.add_solid(s,label = f"{i}")
visApp.redraw() #must be called after adding all solids

visApp.setup_default_camera()
visApp.show_axes()



