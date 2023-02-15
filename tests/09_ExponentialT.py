from bim4loc.random import one_dim as r_1d
import matplotlib.pyplot as plt

p = r_1d.ExponentialT(0.2, 10)
print(p.pdf(10))
p.plot()
plt.show()

# p = r_1d.GaussianT(0, 2, -3, 5)
# p.plot()
# plt.show()
