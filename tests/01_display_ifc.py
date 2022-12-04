from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, IFC_BUILDING_PATH, IFC_SUZANNE_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter
import logging
from bim4loc.solids import Label3D
import numpy as np

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)


solids = ifc_converter(IFC_SUZANNE_PATH)
visApp = VisApp()

for i, s in enumerate(solids):
    visApp.add_solid(s,label = f"{i}")
visApp.redraw() #must be called after adding all solids

visApp.setup_default_camera()
visApp.show_axes()



