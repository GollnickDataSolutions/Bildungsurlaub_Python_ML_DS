#%%
from sklearn.datasets import fetch_openml
import pandas as pd
#%%
data = fetch_openml('adult', version=2, as_frame=True)
# %%
data
# %%
data.keys()
# %%
data["DESCR"]
# %%
X, y = data["data"], data["target"]

#%% Spalten löschen
print(X.isna().sum())  # Überblick über Spalten mit Nan
X.drop(columns=["workclass", "occupation", "native-country"], inplace=True)

#%%
pd.get_dummies(X, dtype=int, drop_first=True)

#%%
# y ["<=50K", ">50K"]
# y [0, 1]
# y = [0 if i=="<=50K" else 1 for i in y]
# %%
