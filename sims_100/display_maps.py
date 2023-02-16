import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from matplotlib.ticker import FormatStrFormatter
import os
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter

seed_number = 10

out_folder = 'out_mid_noise'
dir_path = os.path.dirname(os.path.realpath(__file__))
data_file = os.path.join(dir_path, out_folder , f"data_{seed_number}.p")
data = pickle.Unpickler(open(data_file, "rb")).load()



solids = ifc_converter(data['IFC_PATH'])
visApp = VisApp()

for i, s in enumerate(solids):
    if s.name in data['ground_truth']['constructed_solids_names']:
        visApp.add_solid(s)
visApp.redraw() #must be called after adding all solids
visApp.setup_default_camera()
visApp.show_axes()

