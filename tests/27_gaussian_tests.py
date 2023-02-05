from bim4loc.random.one_dim import Gaussian, Uniform,  ExponentialT
import numpy as np
import matplotlib.pyplot as plt
g = Gaussian(0,1)

v = g.sample(5000)
d = Uniform(0,10)
d_v = d.sample(1000)
r = ExponentialT(1,1)
r_v = r.sample(1000)


# print(g.Anderson_Darling(v))
print(g.Shapiro_Wilk(v))
print(g.Shapiro_Wilk(d_v))
print(g.Shapiro_Wilk(r_v))
# plt.hist(v, bins=100)
# plt.show()

