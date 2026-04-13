# %%
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt
 
# creation of data
n, dim = 500, 1  # number of samples, dimension
n_bkps, sigma = 3, 1  # number of change points, noise standart deviation
signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)
 
# change point detection
model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
algo = rpt.Window(width=40, model=model).fit(signal)
my_bkps = algo.predict(n_bkps=3)
 
# show results
rpt.show.display(signal, bkps, my_bkps, figsize=(10, 6))
#rpt.show.display(signal, my_bkps, figsize=(10, 6))
plt.show()
# %%
 