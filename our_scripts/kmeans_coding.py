#%% Pakete 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from plotnine import ggplot, aes, geom_point, labs
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

#%% Daten vorbereiten
X, _ = make_blobs(n_samples = 1000, centers=3, cluster_std = 2.5, n_features=2, random_state=42)
df_X = pd.DataFrame(X, columns=["x1", "x2"])
df_X

ggplot(df_X) + aes(x="x1", y="x2") + geom_point() 
# %% Elbow diagram
heterogeneity = []
cluster_number = []
for n_clusters in range(1, 8):
    model = KMeans(n_clusters=n_clusters)
    model.fit(X)
    cluster_number.append(n_clusters)
    heterogeneity.append(model.inertia_)

sns.lineplot(x=cluster_number, y=heterogeneity)
plt.show()

#%% mathematische (automatisierte) Ermittlung der opt. Clusteranzahl
scores = {}
for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42).fit(X)
    scores[k] = silhouette_score(X, model.labels_)

best_k = max(scores, key=scores.get)

# %% Modelling
model = KMeans(n_clusters=3)
model.fit(X)


df_X['cluster'] = model.predict(X)
df_X
ggplot(df_X) + aes(x="x1", y="x2",color="cluster") + geom_point() 
# %%
