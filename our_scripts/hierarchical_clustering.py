#%% Pakete
# data handling
import numpy as np
import pandas as pd
import random

# modeling
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

# visualisation
from plotnine import ggplot, aes, geom_point, geom_text
import matplotlib.pyplot as plt
#%% 
num_points = 10
x = random.sample(population=list(set(np.linspace(start=0, stop=10, num=num_points))), k=num_points)
y = random.sample(population=list(set(np.linspace(start=0, stop=10, num=num_points))), k=num_points)
labels = range(1, num_points+1)
df = pd.DataFrame(list(zip(x, y, labels)), columns=['x', 'y', 'point_labels'])


# %%
(ggplot(data=df)
  + aes(x='x', y='y', label='point_labels')
  + geom_point(size=0)
  + geom_text(size=20)
)

#%%
X = np.array(df.drop(columns=["point_labels"]))
X
#%% 
linked = linkage(X, method="centroid", metric="euclidean")
# %%
plt.figure(figsize=(7,7))
dendrogram(linked, labels=labels)
plt.show()
# %% Clustering Modell
model = AgglomerativeClustering(n_clusters=3, metric = "euclidean", linkage="ward")
df["cluster_pred"] = model.fit_predict(X)


# %%
(ggplot(data=df)
  + aes(x='x', y='y', label='point_labels', color='cluster_pred')
  + geom_point(size=0)
  + geom_text(size=20)
)