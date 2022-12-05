import numpy as np
from bim4loc.random.multi_dim import SigmaPoints

# mu = np.zeros(2)
# cov = np.eye(2)
# sigmapoints = SigmaPoints(2, alpha = 1.0, beta = 2.0, mu = mu, cov = cov)
# print(sigmapoints.weights)
# print(sigmapoints.points)

# mu = np.zeros(3)
# cov = np.eye(3)
# sigmapoints = SigmaPoints(3, alpha = 1.2, beta = 2.0, mu = mu, cov = cov)
# print(sigmapoints.weights)
# print(sigmapoints.points)

mu = np.zeros(4)
cov = np.eye(4)
sigmapoints = SigmaPoints(4, alpha = 1.2, beta = 2.0, mu = mu, cov = cov)
print(sigmapoints.weights)
print(sigmapoints.points)

