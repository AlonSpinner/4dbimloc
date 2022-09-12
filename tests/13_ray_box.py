import numpy as np
from bim4loc.geometry.raycaster import ray_box_intersection

box = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
ray = np.array([-3.0, 0.0, 0.0, 1.0, 0.0, 0.0])
inv_ray = 1/ray[3:]

print(ray_box_intersection(ray[:3], inv_ray, box))