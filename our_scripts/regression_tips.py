#%% Paket
import seaborn as sns
import pandas as pd

#%%
tips = sns.load_dataset("tips")
#%%
# size...Anzahl Personen
tips
# %% Datenimport
tips_dummies = pd.get_dummies(data=tips, dtype=int, drop_first=True)
tips_dummies
#%% Exploratory Data Analysis
sns.heatmap(tips[["total_bill", "size", "tip"]].corr(), annot=True)

#%%
tips.describe()
tips.info()

#%% X, y

#%% train, test Dataframes erstellen

#%% Skalierung durchführen

#%% Modell trainieren

#%% Für Testdaten Vorhersagen erstellen

#%% Modellevaluierung auf Basis der Testdaten durchführen (R^2)
