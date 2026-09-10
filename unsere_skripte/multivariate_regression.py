#%% Pakete
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_line
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
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

#%% Trainings- und Testdaten trennen (Zeilen trennen)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

#%% Standardisierung
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Modellinstanz erstellen
model = LinearRegression()

#%% Modell trainieren (fit)
model.fit(X_train_scaled, y_train)

#%%
model.coef_

#%% Vorhersagen erstellen (predict)
y_test_pred = model.predict(X_test_scaled)
sns.regplot(x=y_test, y=y_test_pred)

#%% Evaluierung des Modells durchführen
r2 = r2_score(y_test, y_test_pred)
# %%
