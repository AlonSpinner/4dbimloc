from bim4loc.geometry import pose2z

import numpy as np

u = np.array([1,0,0,0])
COV = np.diag([0.5,0.1,1e-25,np.radians(1)])
x0 = np.array([0,0,0,0])
x1 = pose2z.compose_s(x0, u)
x2 = pose2z.compose_s(x1, u)
x3 = pose2z.compose_s(x2, u)

#adj maps from tangent vectors in pose to tangent vectors in origin
#for this reason, the adjoint can maps covariance matrices, COVe = adj COVx adj.T
#we want the covariance at the last pose, in the tangent space of the last pose.
#so at each conesecutive pose, we need to "drag" the covairance with us, and add COV to it
#to drag a covairnace to a new pose, we use the inverse adjoint.
inv_adj_u = np.linalg.inv(pose2z.adjoint(u))
COV1 = COV
COV2 = inv_adj_u @ COV1 @ inv_adj_u.T + COV
COV3 = inv_adj_u @ COV2 @ inv_adj_u.T + COV

print(COV3)

