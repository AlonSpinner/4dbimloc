# https://stackoverflow.com/questions/62861868/how-to-draw-a-sample-from-a-categorical-distribution

import numpy as np
import matplotlib.pyplot as plt
import pymc3

np.random.seed(25)


N_categories = 10
N_samples = 10
categories = np.arange(N_categories)
p_categories = np.random.uniform(0,1, N_categories)
p_categories = p_categories/np.sum(p_categories) #normalize
cum_prob = np.cumsum(p_categories, axis = 0) #keeps shape

def plot_compare(samples, sampler_name):
    fig, ax = plt.subplots()
    ax.bar(categories, p_categories, color = 'blue', align = 'center')
    ax.hist(samples, bins = categories, density = True, color = [1.0,0.3,0.3,0.5], align = 'left')
    ax.set_title(sampler_name)
    ax.legend(['distribution','samples'])
    # ax.set_ylim([0,0.3])
# --------------------------------------------------------------------------------------------------------
# -------------------------------------- NAIVE DISCRETE EVENT SAMPLER-------------------------------------
# --------------------------------------------------------------------------------------------------------


samples = np.zeros(N_samples)

r = np.random.uniform(0, 1, N_samples)
for i, ri in enumerate(r):
    samples[i] = np.searchsorted(cum_prob, ri)
plot_compare(samples, "Naive discrete event sampler")


# --------------------------------------------------------------------------------------------------------
# -------------------------------------- LOW VARIANCE SAMPLER---------------------------------------------
# --------------------------------------------------------------------------------------------------------

N_samples_LVS = int(N_samples/N_categories) * N_categories
samples = np.zeros(N_samples_LVS)

for k in range(int(N_samples_LVS/N_categories)):
    r = np.random.uniform()/N_categories
    idx = 0
    c = p_categories[0]
    for i in range(N_categories):
        uu = r + i*1/N_categories
        while uu > c:
            idx += 1
            c += p_categories[idx]
        samples[i + k*N_categories] = categories[idx]

plot_compare(samples, "Low Variance Sampler")

# --------------------------------------------------------------------------------------------------------
# -------------------------------------- MultiNomial---------------------------------------------
# --------------------------------------------------------------------------------------------------------

hist = np.random.multinomial(N_samples, p_categories)
samples = []
for i, h in enumerate(hist):
    samples.extend([i] * h)
samples = np.array(samples)


plot_compare(samples, "multinomial")
# --------------------------------------------------------------------------------------------------------
# -------------------------------------- PYMC3---------------------------------------------
# --------------------------------------------------------------------------------------------------------

#https://discourse.pymc.io/t/preferred-sampler-for-categorical-predictors/7229/2
#https://stats.stackexchange.com/questions/171269/pymc3-sampling-from-a-categorical-distribution
# https://www.statlect.com/fundamentals-of-statistics/Metropolis-Hastings-algorithm
# with pymc3.Model() as model:
#     category = pymc3.Categorical(name='category',
#                                  p = p_categories)
#     start = pymc3.find_MAP()
#     step = pymc3.NUTS(scaling=start)
#     trace = pymc3.sample(20, start = start, step=step)
# samples = trace['category']
# plot_compare(samples, "pymc3")


plt.show()