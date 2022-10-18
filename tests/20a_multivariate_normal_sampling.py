import numpy as np
from bim4loc.random.multi_dim import sample_normal
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt



mu = np.array([1,2])
cov = np.array([[1.0, 0.5], 
                [0.5, 1.0]])

z = sample_normal(mu, cov, 1000)
sns.jointplot(x = z[:,0], y = z[:,1], kind = 'kde', space = 0, shade = True)
plt.show()


