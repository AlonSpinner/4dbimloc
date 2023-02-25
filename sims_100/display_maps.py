import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from matplotlib.ticker import FormatStrFormatter
import os
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter

seeds = range(5)
def crop_image(image, crop_ratio_w, crop_ratio_h):
    h,w = image.shape[:2]
    crop_h = int(h * crop_ratio_h/2)
    crop_w = int(w * crop_ratio_w/2)
    return image[crop_h:-crop_h, crop_w:-crop_w,:]
dir_path = os.path.dirname(os.path.realpath(__file__))
images_output_path = os.path.join(dir_path, "maps")
data_folder = 'out_mid_noise'

for seed_number in seeds:
    data_file = os.path.join(dir_path, data_folder , f"data_{seed_number}.p")
    data = pickle.Unpickler(open(data_file, "rb")).load()
    solids = ifc_converter(data['IFC_PATH'])
    visApp = VisApp()
    for i, s in enumerate(solids):
        if s.name in data['ground_truth']['constructed_solids_names']:
            visApp.add_solid(s)
    visApp.redraw() #must be called after adding all solids
    visApp.setup_default_camera()
    visApp.show_axes(False)

    images = visApp.get_images(images_output_path,prefix = f"M{seed_number}_",
                    transform = lambda x: crop_image(x,0.6,0.55), save_scenes = ["world"])
    visApp.quit()

print('finished')

