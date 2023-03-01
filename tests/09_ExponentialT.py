from bim4loc.random import one_dim as r_1d
import matplotlib.pyplot as plt

max_range = 20
p = r_1d.ExponentialT(0.4, max_range)
p.plot()

max_range = 6
p = r_1d.ExponentialT(0.4, max_range)
p.plot()
plt.show()

# p = r_1d.GaussianT(0, 2, -3, 5)
# p.plot()
# plt.show()
