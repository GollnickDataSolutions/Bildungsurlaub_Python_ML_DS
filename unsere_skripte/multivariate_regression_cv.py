#%% Pakete
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_line
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

# %% Daten importieren
file_path = "../001_Datasets/winequality-red.csv"
wine = pd.read_csv(file_path, sep=";")
wine
# %% EDA
wine.head()

#%%
wine.dtypes
# %%
wine.describe()

#%%
sns.pairplot(wine.iloc[:, 7:12], hue='quality')
# %%
sns.heatmap(wine.corr(), annot=True)
plt.savefig("wine_quality_analysis.png")

plt.show()
# %% Unabhängige und abhängige Variable trennen (Spalten trennen)
X = wine.drop(columns=['quality'])
y = wine['quality']
print(f"wine shape: {wine.shape}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

#%% K-Fold CV
kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold CV

#%%
r2_scores = []

for train_idx, test_idx in kf.split(X):
    model = LinearRegression()
    # print(f"Train Indices: {train_idx}")
    # print(f"Test Indices: {test_idx}")
    X_train = X.iloc[train_idx, :]
    X_test = X.iloc[test_idx, :]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    
    model.fit(X_train, y_train)  # Modell trainieren
    r2 = model.score(X_test, y_test)
    r2_scores.append(r2)
    print(f"R2 score: {r2}")

#%% Ergebnisse der Validierung mitteln
np.mean(r2_scores)

#%% Modellinstanz erstellen
model = LinearRegression()

#%% Modell trainieren (fit)
model.fit(X_train, y_train)

#%%
model.coef_

#%% Vorhersagen erstellen (predict)
y_test_pred = model.predict(X_test)
sns.regplot(x=y_test, y=y_test_pred)

#%% Evaluierung des Modells durchführen
r2 = r2_score(y_test, y_test_pred)
# %%
