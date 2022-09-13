import numpy as np
from bim4loc.geometry import pose2z

x0 = np.array([3.0, 3.0, 1.5, np.pi])
dx = np.array([0.5, 0.0, 0.0, 0.0])

print(pose2z.R_from_theta(x0[3]))

x1 = pose2z.compose_s(x0, dx)
print(x1)