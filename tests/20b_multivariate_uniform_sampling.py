import numpy as np
from bim4loc.random.multi_dim import sample_uniform
import time
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt



a = np.zeros(2)
b = np.ones(2)

z = sample_uniform(a, b, 1000)
# sns.jointplot(x = z[:,0], y = z[:,1], kind = 'kde', space = 0)
# plt.show()

s = time.time()
for _ in range(1000000):
    z = sample_uniform(a, b, 1000)
e = time.time()
my_uniform = e - s

s = time.time()
for _ in range(1000000):
    z = np.random.uniform(a, b, (1000,2))
e = time.time()
np_uniform = e - s

print(f"my_uniform/np_uniform = {my_uniform/np_uniform}")
#scores 0.66




