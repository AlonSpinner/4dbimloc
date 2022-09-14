from matplotlib import cm
import numpy as np


my_stupid_array = np.linspace(0,10,100)
cmmap = cm.get_cmap('viridis')
cmmap.get_over()
print(cmmap(my_stupid_array))
