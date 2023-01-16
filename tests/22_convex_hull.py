from bim4loc.geometry.minimal_distance import convex_hull, T_from_pitch_yaw
import bim4loc.geometry.s03 as s03
import numpy as np
import matplotlib.pyplot as plt


def distance_to_line(p0,p1,q):
    #p0 and p1 are two points on the line
    #q is the point we want to find the distance to
    #https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    projected_point = p0 + np.dot(q-p0,p1-p0)/np.dot(p1-p0,p1-p0)*(p1-p0)
    distance = np.linalg.norm(projected_point-q)
    return distance, projected_point

def point_to_convex_hull_dist(point, hull):
    s = np.inf
    projected_point = np.zeros(2)
    hull_plus = np.vstack((hull,hull[0]))
    for i in range(1,hull.shape[0]):
        test_s, test_projected_point =  distance_to_line(hull_plus[i-1], hull_plus[i], point)
        if test_s < s:
            s = test_s
            projected_point = test_projected_point
    return s, projected_point

points = np.random.uniform(-np.pi/2,np.pi/2,(30,2))   # 30 random points in 2-D
points = np.hstack((np.zeros((points.shape[0],1)), points))
rots = [s03.hat(p) for p in points]
rot_bar = s03.mu_rotations(rots)

query = np.array([0, np.radians(0),np.radians(30)])
rot_query = s03.hat(query)

q = s03.vee(s03.minus(rot_bar,rot_query))
p = [s03.vee(s03.minus(rot_bar,rot)) for rot in rots]

hull = convex_hull(np.array(p))
hull_plus = np.vstack((hull,hull[0]))
s, projected_point = point_to_convex_hull_dist(q, hull)

hull = convex_hull(points)
hull_plus = np.vstack((hull,hull[0]))
projected_point = s03.vee(s03.plus(rot_bar, s03.hat(projected_point)))

plt.plot(points[:,0], points[:,1], 'o')
plt.plot(hull_plus[:,0], hull_plus[:,1], 'r--', lw=2)
plt.plot(query[0], query[1], 'ro')
plt.plot([query[0], projected_point[0]], [query[1], projected_point[1]], 'k--')
plt.axis('equal')
plt.show()








