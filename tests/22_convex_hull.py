from bim4loc.geometry.minimal_distance import convex_hull, T_from_pitch_yaw
import numpy as np
import matplotlib.pyplot as plt

def vee(Q):
    return np.array([Q[2,1],Q[0,2],Q[1,0]])

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
        print(i)
    return s, projected_point

points = np.random.uniform(-np.pi/2,np.pi/2,(30,2))   # 30 random points in 2-D
hull = convex_hull(points)
print(hull)

query = np.array([np.radians(0),np.radians(30)])
T = T_from_pitch_yaw(query[0], query[1])
R = T[:3,:3]

t = np.arccos((np.trace(R)-1)/2)
theta = t * vee(R - R.T)/(2*np.sin(t))
print(np.degrees(theta))

hull_plus = np.vstack((hull,hull[0]))
s, projected_point = point_to_convex_hull_dist(query, hull)

plt.plot(points[:,0], points[:,1], 'o')
plt.plot(hull_plus[:,0], hull_plus[:,1], 'r--', lw=2)
plt.plot(query[0], query[1], 'ro')
plt.plot([query[0], projected_point[0]], [query[1], projected_point[1]], 'k--')
plt.axis('equal')
plt.show()








