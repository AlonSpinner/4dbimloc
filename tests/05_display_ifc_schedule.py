from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter
from bim4loc.random.one_dim import Gaussian

# import open3d as o3d
# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

solids = ifc_converter(IFC_ONLY_WALLS_PATH)
visApp = VisApp()
current_time = 5.0

for s in solids:
    if isinstance(s.schedule,Gaussian): #should be all of 'em
        s.set_random_completion_time()
        s.set_existance_belief_and_shader(s.schedule.cdf(current_time))
        visApp.add_solid(s)
        
        # if s.is_complete(current_time):
        #         visApp.add_solid(s)
        
visApp.redraw()
# visApp.show_axes()
visApp.setup_default_camera()

imgs = visApp.get_images()
import matplotlib.pyplot as plt
plt.imshow(imgs['world'])
# visApp.redraw()



