from bim4loc.random import one_dim as r_1d
import matplotlib.pyplot as plt

p = r_1d.ExponentialT(0.2, 20)
p.plot()
plt.show()
