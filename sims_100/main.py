from data_collection import create_data
from run_simulations import run_simulation
from video_images_maker import make_video

import numpy as np
for i in range(0,1):
    create_data(i, "data", vis_on = False)
    run_simulation(i, "data" ,"results", vis_on = False)
    make_video(i, "data", "results", "media", save_images = False)
    print(f'finished {i}')
print('done')
