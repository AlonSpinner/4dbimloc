from bim4loc.geometry.minimal_distance import minimal_distance_from_projected_boundry
from bim4loc.geometry.convex_hull import convex_hull_jarvis as convex_hull
from bim4loc.geometry.utils import point_in_polygon
import numpy as np
import matplotlib.pyplot as plt

base_element_angle = np.pi * -1
d_base_element_angle = np.pi/6

projected_verts = np.random.uniform(base_element_angle - d_base_element_angle,
                                    base_element_angle + d_base_element_angle,
                                    (60,2))
ray_point = np.array([base_element_angle + np.random.uniform(0,d_base_element_angle/2),
                     base_element_angle + np.random.uniform(0,d_base_element_angle/2)])
s, dq = minimal_distance_from_projected_boundry(ray_point, projected_verts)

projected_point = ray_point + dq

hull = convex_hull(projected_verts)
hull_plus = np.vstack((hull,hull[0]))

plt.plot(projected_verts[:,0], projected_verts[:,1], 'o')
plt.plot(hull_plus[:,0], hull_plus[:,1], 'r--', lw=2)
plt.plot(ray_point[0], ray_point[1], 'ro')
plt.plot([ray_point[0], projected_point[0]], [ray_point[1], projected_point[1]], 'k--')
if point_in_polygon(ray_point, hull_plus):
    plt.title('Point is inside the convex hull')
else:
    plt.title('Point is outside the convex hull')
plt.axis('equal')
print(s)
plt.show()








