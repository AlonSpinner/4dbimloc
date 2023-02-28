from data_collection import create_data
from run_simulations import run_simulation
from video_images_maker import make_video

for i in range(94,100):
    create_data(i, "data", vis_on = False)
    run_simulation(i, "data" ,"results", vis_on = False)
    print(f'finished {i}')
print('done')

for i in range(0,100):
    make_video(i, "data", "results", "media", save_images = False)
    print(f'finished {i}')
print('done')
