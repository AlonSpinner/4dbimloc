from matplotlib import cm
import numpy as np


colors = cm.get_cmap('plasma', 20)(np.linspace(0, 1, 20))

colors = ["#084594", "#0F529E", "#1760A8", "#1F6EB3", "#2979B9", "#3484BE", "#3E8EC4",
              "#4A97C9", "#57A0CE", "#64A9D3", "#73B2D7", "#83BBDB", "#93C4DE", "#A2CBE2",
              "#AED1E6", "#BBD6EB", "#C9DCEF", "#DBE8F4", "#EDF3F9", "#FFFFFF"]
colors = np.array(colors)
v = np.linspace(0,10,100)
c = ((v-v.min())/(v.max()-v.min())*(len(colors)-1)).astype(np.int16)


cmmap = cm.get_cmap('viridis')
cmmap.get_over()
print(cmmap(my_stupid_array))
