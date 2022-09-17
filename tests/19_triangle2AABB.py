import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d
import matplotlib.colors as colors
from bim4loc.geometry.raycaster import triangle_to_AABB

N_triangles = 10

world_box = np.array([-1.0, -1.0, -1.0, +1.0, +1.0, +1.0])
p1 = np.random.uniform(world_box[:3], world_box[3:], (N_triangles,3))
p2 = np.random.normal(p1, 0.2, (N_triangles,3))
p3 = np.random.normal(p2, 0.2, (N_triangles,3))

triangles = np.stack((p1,p2,p3), axis = 2).reshape(N_triangles,3,3).transpose(0,2,1)
#triangles are set up as:
# [[p1],
# [p2],
# [p3]] (nx3)

def plot_triangle(ax : plt.Axes, triangle : np.ndarray):
     tri = mplot3d.art3d.Poly3DCollection([triangle]) #vertices are nx3
     tri.set_color(colors.rgb2hex(np.random.rand(3)))
     tri.set_edgecolor('k')
     ax.add_collection3d(tri)

def plot_2_points(ax : plt.Axes, p0 ,p1):
    p = np.vstack([p0,p1]).T
    ax.plot(p[0],p[1],p[2], color = 'k')

def plot_AABB(ax: plt.Axes, AABB : np.ndarray):
    p_max = AABB[3:]
    p_min = AABB[:3]

    cube_bottom = np.zeros((4,3))
    cube_bottom[:,0] = np.array([p_min[0],p_max[0],p_max[0],p_min[0]])
    cube_bottom[:,1] = np.array([p_min[1],p_min[1],p_max[1],p_max[1]])
    cube_bottom[:,2] = np.array([p_min[2],p_min[2],p_min[2],p_min[2]])

    cube_top = np.zeros((4,3))
    cube_top[:,0] = np.array([p_min[0],p_max[0],p_max[0],p_min[0]])
    cube_top[:,1] = np.array([p_min[1],p_min[1],p_max[1],p_max[1]])
    cube_top[:,2] = np.array([p_max[2],p_max[2],p_max[2],p_max[2]])

    plot_2_points(ax, cube_bottom[0], cube_bottom[1])
    plot_2_points(ax, cube_bottom[1], cube_bottom[2])
    plot_2_points(ax, cube_bottom[2], cube_bottom[3])
    plot_2_points(ax, cube_bottom[3], cube_bottom[0])

    plot_2_points(ax, cube_bottom[0], cube_top[0])
    plot_2_points(ax, cube_bottom[1], cube_top[1])
    plot_2_points(ax, cube_bottom[2], cube_top[2])
    plot_2_points(ax, cube_bottom[3], cube_top[3])

    plot_2_points(ax, cube_top[0], cube_top[1])
    plot_2_points(ax, cube_top[1], cube_top[2])
    plot_2_points(ax, cube_top[2], cube_top[3])
    plot_2_points(ax, cube_top[3], cube_top[0])

fig = plt.figure()
ax = plt.axes(projection='3d')
for t in triangles:
    plot_triangle(ax, t)
    plot_AABB(ax,triangle_to_AABB(t))


xs = triangles[:,:,0]; ax.set_xlim(np.min(xs),np.max(xs))
ys = triangles[:,:,1]; ax.set_ylim(np.min(ys),np.max(ys))
zs = triangles[:,:,2]; ax.set_zlim(np.min(zs),np.max(zs))

plt.show()
