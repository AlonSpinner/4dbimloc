from bim4loc.geometry.minimal_distance import convex_hull, T_from_pitch_yaw
import numpy as np

points = np.random.uniform(0,1,(30,2))   # 30 random points in 2-D
hull = convex_hull(points)
print(hull)

T = T_from_pitch_yaw(np.radians(90),np.radians(90))
R = T[:3,:3]

def vee(Q):
    return np.array([Q[2,1],Q[0,2],Q[1,0]])
t = np.arccos((np.trace(R)-1)/2)
theta = t * vee(R - R.T)/(2*np.sin(t))
print(np.degrees(theta))






