from data_collection import create_data
from run_simulations import run_simulation
from video_images_maker import make_video
from do_statistical_analysis import statistical_analysis
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
out_folder  = os.path.join(dir_path,"out1")
# for i in range(4,30):
#     create_data(i, out_folder, vis_on = False)
#     run_simulation(i, out_folder, vis_on = False)
#     print(f'finished {i}')
statistical_analysis(out_folder, range(30))

print('finished out1')
out_folder  = os.path.join(dir_path,"out2")
for i in range(0,30):
    create_data(i, out_folder, vis_on = False)
    run_simulation(i, out_folder, vis_on = False)
    print(f'finished {i}')
statistical_analysis(out_folder, range(30))

# for i in range(0,30):
#     make_video(i, out_folder, save_images = False)
#     print(f'finished {i}')
# print('done')
