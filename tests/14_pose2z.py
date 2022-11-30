import numpy as np
from bim4loc.geometry import pose2z

x0 = np.array([3.0, 3.0, 1.5, np.pi])
dx = np.array([0.5, 0.0, 0.0, 0.0])

print(pose2z.R_from_theta(x0[3]))

x1 = pose2z.compose_s(x0, dx)
print(x1)

print(pose2z.compose_s_array(np.vstack((x0,x0)), dx))


x = np.array([1.0,0,0,0])
p_world = np.array([[2.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0]]).T
pose2z.transform_from(x, p_world)
pose2z.transform_to(x, p_world)
theta = pose2z.angle(x, p_world)
print(theta)
