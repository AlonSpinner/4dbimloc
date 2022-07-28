from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.ifc import converter
import time

objects = converter(IFC_ONLY_WALLS_PATH)
visApp = VisApp()

for o in objects:
    visApp.add_object(o)