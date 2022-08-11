from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH, IFC_TEST_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter
from bim4loc.random_models.one_dim import Gaussian

# import open3d as o3d
# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

solids = ifc_converter(IFC_ONLY_WALLS_PATH)
visApp1 = VisApp()
current_time = 5.0

for s in solids:
    if isinstance(s.schedule,Gaussian): #should be all of 'em
        s.set_random_completion_time()
        # if s.is_complete(current_time):
                # visApp1.add_solid(s)
        s.set_existance_belief_by_schedule(current_time, set_shader = True)
        
[visApp1.add_solid(s) for s in solids]

#show axes and reset camera after building scene
visApp1.show_axes(True)
visApp1.reset_camera_to_default()



