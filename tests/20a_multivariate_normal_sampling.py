import numpy as np
from bim4loc.random.multi_dim import sample_normal
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
import time


mu = np.array([1,2])
cov = np.array([[1.0, 0.5], 
                [0.5, 1.0]])

z = sample_normal(mu, cov, 1000)
# sns.jointplot(x = z[:,0], y = z[:,1], kind = 'kde', space = 0)
# plt.show()

z = sample_normal(mu, cov, 1000)
s = time.time()
for _ in range(10000):
    z = sample_normal(mu, cov, 1000)
e = time.time()
my_time = e - s

s = time.time()
for _ in range(10000):
    z = np.random.multivariate_normal(mu, cov, 1000)
e = time.time()
np_time = e - s

print(f"my_time/np_time = {my_time/np_time}")
#scores 1.209


