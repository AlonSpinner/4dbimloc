# this guy is a god:
#https://ethanweed.github.io/pythonbook/05.02-ttest.html

from pingouin import ttest
import pandas as pd
import numpy as np
x = np.arange(10)
y = np.arange(10)
df = pd.DataFrame(data = {'x': x, 'y': y})
results = ttest(df['x'], df['y'], paired=True, alternative = 'greater')

a = 2