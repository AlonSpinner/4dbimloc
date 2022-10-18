import numpy as np
from bim4loc.random.multi_dim import sample_uniform
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt



a = np.zeros(2)
b = np.ones(2)

z = sample_uniform(a, b, 1000)
sns.jointplot(x = z[:,0], y = z[:,1], kind = 'kde', space = 0, shade = True)
plt.show()


