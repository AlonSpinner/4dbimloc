import numpy as np
import matplotlib.pyplot as plt

x = 1
rate = 2

t = np.arange(4)
history = np.zeros_like(t,dtype = float)
history[0] = x
for i in range(1,len(t)):
    x = x * np.exp(-rate)
    history[i] = x
plt.plot(t, history)
plt.show()