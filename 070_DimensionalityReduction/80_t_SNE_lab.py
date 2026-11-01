# %% tSNE
# # Introduction
# %% [markdown]
# t-Distributed Stochastic Neighbor Embedding (t-SNE) is an unsupervised, non-linear technique primarily used for data exploration and visualizing high-dimensional data. In simpler terms, t-SNE gives you a feel or intuition of how the data is arranged in a high-dimensional space. It was developed by Laurens van der Maatens and Geoffrey Hinton in 2008. [source](https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1)
# %% [markdown]
# # Packages

# %%
import pandas as pd
from plotnine import ggplot, aes, geom_point # ggplot
from sklearn import datasets  # Mnist dataset
from sklearn.manifold import TSNE 
import seaborn as sns
import matplotlib.pyplot as plt
# %% [markdown]
# # Data Preparation
# %% [markdown]
# We will use Mnist digital character dataset.

# %%
digits = datasets.load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target
X.shape
y.shape
#%% 
digits.images.shape
sns.heatmap(digits.images[0])
plt.show()


# %%
digits_embedded = TSNE(n_components=2).fit_transform(X)

#%% PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit_transform(X)
df_pca = pd.DataFrame(pca, columns=['Component1', 'Component2'])
df_pca['target'] = pd.Categorical(y)

# %%
(ggplot(data=df_pca) 
 + aes(x='Component1', y='Component2', color='target')
 + geom_point()
)

# %%
df = pd.DataFrame(digits_embedded, columns=['Component1', 'Component2'])
df['target'] = pd.Categorical(y)


# %%
(ggplot(data=df_pca ) 
 + aes(x='Component1', y='Component2', color='target')
 + geom_point()
)

# t-SNE is able to correctly separate the digits




# %%
