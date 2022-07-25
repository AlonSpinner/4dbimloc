from bim4loc.binaries.paths import IFC_SIMPLE_PATH, DRONE_PATH, IFC_TEST_PATH
from bim4loc.ifc import converter
import trimesh

objects = converter(IFC_TEST_PATH)
t = 4.0 #current time

objects[0].geometry