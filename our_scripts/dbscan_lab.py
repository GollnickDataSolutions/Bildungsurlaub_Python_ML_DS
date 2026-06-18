#%% Pakete
# data handling
import numpy as np
import pandas as pd
import random

# modeling
from sklearn.cluster import DBSCAN

# visualisation
from plotnine import ggplot, aes, geom_point, geom_text
import matplotlib.pyplot as plt
# %%
# %% Data Preparation
num_points = 4000
x = random.sample(population=list(set(np.linspace(start=-10, stop=10, num=num_points))), k=num_points)
y = random.sample(population=list(set(np.linspace(start=-10, stop=10, num=num_points))), k=num_points)
z = [(xi**2 + yi**2) for xi, yi in zip(x, y)]
#%%
df = pd.DataFrame(list(zip(x, y, z)), columns=['x', 'y', 'z'])
df['class'] = [1 if ((i<10) | (80 < i< 100))  else 0 for i in df['z']]

df = df[df['class']==1]  # filter for class 1
df = df.drop(['z', 'class'], axis=1)  # delete not required columns


# %% Visualisation of results
(ggplot(data=df) +
  aes(x='x', y='y') +
  geom_point()
)

# %%
model = DBSCAN(eps=3, min_samples=2)
model.fit(df)

#%%
df["cluster"] = model.labels_
(ggplot(data=df) +
  aes(x='x', y='y', color='cluster') +
  geom_point()
)
# %%
