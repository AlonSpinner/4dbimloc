from bim4loc.geometry.minimal_distance import convex_hull, T_from_pitch_yaw
import bim4loc.geometry.so3 as so3
import numpy as np
import matplotlib.pyplot as plt
from bim4loc.geometry.pca import pca


def distance_to_line(p0,p1,q):
    #p0 and p1 are two points on the line
    #q is the point we want to find the distance to
    #https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    projected_point = p0 + np.dot(q-p0,p1-p0)/np.dot(p1-p0,p1-p0)*(p1-p0)
    distance = np.linalg.norm(projected_point-q)
    return distance, projected_point

def point_to_convex_hull_dist(point, hull_plus):
    #hull_plus = np.vstack((hull,hull[0]))
    s = np.inf
    projected_point = np.zeros(2)
    for i in range(1,hull.shape[0]):
        test_s, test_projected_point =  distance_to_line(hull_plus[i-1], hull_plus[i], point)
        if test_s < s:
            s = test_s
            projected_point = test_projected_point
    return s, projected_point

projected_verts = np.random.uniform(np.pi - np.pi/6,np.pi + np.pi/6,(30,2))   # 30 random points in 2-D
projected_verts = np.hstack((np.zeros((projected_verts.shape[0],1)), projected_verts)) #roll-pitch-yaw
rots = [so3.exp(p) for p in projected_verts]
rot_bar = so3.mu_rotations(rots)

query = np.array([0, np.radians(0),0])
rot_query = so3.exp(query)

d_query = so3.log(so3.minus(rot_bar,rot_query))
d_vertices = [so3.log(so3.minus(rot_bar,rot)) for rot in rots]

#because the convex hull is in 2D, we need to project the vertices onto the plane
#defined by the rotation rot_bar
points = np.array([so3.log(so3.minus(rot_bar,rot)) for rot in rots])
p = np.array([so3.log(so3.minus(rot_bar,rot)) for rot in rots])
q = so3.log(so3.minus(rot_bar,rot_query))

phat, transform = pca(p)
qhat = q @ transform

hull = convex_hull(np.array(phat))
hull_plus = np.vstack((hull,hull[0]))
s, projected_point = point_to_convex_hull_dist(qhat, hull_plus)

plt.plot(phat[:,0], phat[:,1], 'o')
plt.plot(hull_plus[:,0], hull_plus[:,1], 'r--', lw=2)
plt.plot(qhat[0], qhat[1], 'ro')
plt.plot([qhat[0], projected_point[0]], [qhat[1], projected_point[1]], 'k--')
plt.axis('equal')
plt.show()








