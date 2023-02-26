import numpy as np
import pickle
import os
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter
from bim4loc.agents import Drone
from bim4loc.maps import RayCastingMap
from bim4loc.geometry import pose2z
from bim4loc.solids import ifc_converter, ParticlesSolid, TrailSolid, ScanSolid, Label3D, \
                            update_existence_dependence_from_yaml, add_variations_from_yaml
import imageio
from PIL import Image, ImageEnhance, ImageDraw, ImageFont

save_images = False
A = 1; B = 2; C = 3; D = 4
variation_names = {0 : "Simulation", A : "BPFS", B : "BPFS-t", C : "BPFS-tg", D : "logodds"}

#get data and results
dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path, "25a_data.p")
data = pickle.Unpickler(open(file, "rb")).load()
file = os.path.join(dir_path, "25b_results.p")
results = pickle.Unpickler(open(file, "rb")).load()
yaml_file = os.path.join(dir_path, "25_complementry_IFC_data.yaml")
output_image_path = os.path.join(dir_path, "25_images")

#BUILD GROUND TRUTH
solids = ifc_converter(data['IFC_PATH'])
world_solids = [s.clone() for s in solids if s.name in data['ground_truth']['constructed_solids_names']]
sensor = data['sensor']
drone = Drone(pose = data['ground_truth']['trajectory'][0])
drone.mount_sensor(sensor)

#ADD TO VISAPP
#WORLD
visApp = VisApp()
[visApp.add_solid(s, "world") for s in world_solids]
visApp.redraw("world")
visApp.setup_default_camera("world")
vis_scan = ScanSolid("scan")
visApp.add_solid(drone.solid,"world")
visApp.add_solid(vis_scan, "world")
trail_ground_truth = TrailSolid("trail_ground_truth", drone.pose[:3].reshape(1,3))
visApp.add_solid(trail_ground_truth, "world")
#METHODS
def add_method2_visApp(N : int, window_name):
    simulation = [s.clone() for s in solids]
    add_variations_from_yaml(simulation, yaml_file)
    update_existence_dependence_from_yaml(simulation, yaml_file)
    simulation = RayCastingMap(simulation)

    scene_name = f"method{N}"
    visApp.add_scene(scene_name, window_name)
    [visApp.add_solid(s,scene_name) for s in simulation.solids]
    visApp.redraw(scene_name)
    visApp.setup_default_camera(scene_name)
    vis_particles = ParticlesSolid(poses = results[N]['particle_poses'][0])
    visApp.add_solid(vis_particles.lines, scene_name)
    visApp.add_solid(vis_particles.tails, scene_name)
    vis_trail_est = TrailSolid("trail_est", results[N]['pose_mu'][0][:3].reshape(1,3))
    visApp.add_solid(vis_trail_est,scene_name)
    # visApp.show_axes(True, scene_name)

    #This isnt saved with images. only for "runtime"
    bmin, bmax, bextent = simulation.bounds()
    locx = (bmin[0]+bmax[0])/2
    locy = bmax[1] + 1.0
    locz = bmax[2]
    visApp.add_text(Label3D(variation_names[N], np.array([locx, locy, locz])), scene_name)
    return simulation, vis_particles, vis_trail_est

def update_method_drawings(N : int, simulation, vis_particles, vis_trail_est):
    scene_name = f"method{N}"
    simulation.update_solids_beliefs(results[N]['expected_belief_map'][t+1])
    [visApp.update_solid(s,scene_name) for s in simulation.solids]
    vis_particles.update(results[N]['particle_poses'][t+1], results[N]['particle_weights'][t+1])
    visApp.update_solid(vis_particles.lines, scene_name)
    visApp.update_solid(vis_particles.tails, scene_name)
    vis_trail_est.update(results[N]['pose_mu'][t+1][:3].reshape(1,3))
    visApp.update_solid(vis_trail_est,scene_name)
    visApp.redraw(scene_name)

def transform_image(image, crop_ratio_w, crop_ratio_h):
        #crop image
        h,w = image.shape[:2]
        crop_h = int(h * crop_ratio_h/2)
        crop_w = int(w * crop_ratio_w/2)
        image  = image[crop_h:-crop_h, crop_w:-crop_w,:]

        #brightness and contrast
        image = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        image = np.array(image)
        return image

def add_text_to_image(image, text, xy_location = None, font_size = 32):
    pillow_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pillow_image)
    font = ImageFont.truetype("arial.ttf", font_size)
    if xy_location is None:
        xy_location = (image.shape[1]//2.5, image.shape[0]//15)

    draw.text(xy_location, text, fill=(0, 0, 0), font=font)
    numpy_image = np.array(pillow_image)
    return numpy_image

def tile_images(images):
    # Get image shape
    image_shape = images[0].shape
    # Create empty canvas to put images
    canvas_shape = (image_shape[0]*2, image_shape[1]*3, image_shape[2])
    canvas = np.ones(canvas_shape,dtype = np.uint8) * 255
    # Tile images onto canvas
    for i in range(5):
        row = i // 3
        col = i % 3
        x = col * image_shape[1]
        y = row * image_shape[0]
        if i == 0:
             y_offset = image_shape[0]//2
             y += y_offset
        if row == 1:
            # Center the images on the bottom row
            x_offset = image_shape[1]
            x += x_offset
        canvas[y:y+image_shape[0], x:x+image_shape[1], :] = images[i].astype(np.uint8)
    return canvas

visApp_A = add_method2_visApp(A,"world")
visApp_B = add_method2_visApp(B,"world")
visApp.add_window("bottom")
visApp_C = add_method2_visApp(C,"bottom")
visApp_D = add_method2_visApp(D,"bottom")
visApp.add_scene("spaceholder", "bottom")

video_canvases = []
for t, z in enumerate(data['measurements']['Z']):
    drone.update_pose(data['ground_truth']['trajectory'][t+1])
    z_p = pose2z.transform_from(drone.pose, drone.sensor.scan_to_points(z))

    #GROUND TRUTH DRAWINGS
    #updating drawings
    vis_scan.update(drone.pose[:3], z_p.T)
    visApp.update_solid(vis_scan)
    visApp.update_solid(drone.solid)
    trail_ground_truth.update(drone.pose[:3].reshape(1,-1))
    visApp.update_solid(trail_ground_truth, "world")
    visApp.redraw("world")
    
    #METHOD DRAWNGS
    update_method_drawings(A, *visApp_A)
    update_method_drawings(B, *visApp_B)
    update_method_drawings(C, *visApp_C)
    update_method_drawings(D, *visApp_D)

    scene_images = visApp.get_images(transform = lambda x: transform_image(x,0.01,0.45))
    scene_images.pop('spaceholder')

    if save_images and t % 3 == 0:
        for i, img in enumerate(scene_images.values()):
            imageio.imwrite(os.path.join(output_image_path,f"t{t}_{variation_names[i]}.png"), img)

    video_images = []
    for i, img in enumerate(scene_images.values()):
        img = add_text_to_image(img, variation_names[i])
        video_images.append(img)
    canvas = tile_images(video_images)
    video_canvases.append(canvas) #drop last scene

imageio.mimsave(os.path.join(dir_path,"25_result_video.mp4"), video_canvases, 'mp4', fps = 10)
visApp.quit()
print("finished")

