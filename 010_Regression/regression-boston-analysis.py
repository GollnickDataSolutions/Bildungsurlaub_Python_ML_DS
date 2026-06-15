#%% Pakete importieren
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from plotnine import ggplot, aes, geom_point, geom_smooth, xlim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import seaborn as sns
#%% Daten einlesen
boston = pd.read_csv("BostonHousing.csv")
boston

#%%
boston.describe()

#%% Paarweiser Vergleich der Spalten
sns.pairplot(boston.iloc[:, :4])
plt.show()

#%% Korrelation der Spalten untereinander
boston_corr = boston.corr()
boston_corr
mask = np.triu(np.ones_like(boston_corr, dtype=bool))
sns.heatmap(boston_corr, mask=mask, annot=True, annot_kws={"size": 6})
plt.show()
#%% Visuelle Prüfung (z.B. x...RM, y...MEDV)
ggplot(boston) + aes(x="rm", y="medv") + geom_point() + geom_smooth(method="lm") + xlim(0, 10)

#%% Modeling
# X und y müssen die Form (n, 1) bzw. (n,)
# X=boston[['rm', 'crim', '...']]  # whitelisting ansatz
X = boston.drop(columns="medv")  # blacklisting ansatz
y = boston[['medv']]

print(X.shape, y.shape)

#%% Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

#%% Standardisierung
# Parameter werden auf Basis der Trainingsdaten ermittelt (fit_transform)
# Die Parameter der Trainingsdaten werden auf die Testdaten angewandt (transform)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#%% Modell fitten
model = LinearRegression()
model.fit(X_train_scaled, y_train)

#%% Parameter anzeigen
print("Schnittpunkt mit y-Achse:", model.intercept_)
print("Steigungswert:", model.coef_)

#%% Vorhersagen erstellen
y_train_pred = model.predict(X_train_scaled)  # (Zeilenanz, 1)  
df_result = y_train
df_result['medv_pred'] = y_train_pred
df_result

ggplot(data=df_result) + aes(x="medv", y="medv_pred") + geom_point() + geom_smooth(method="lm")

#%% Visualisierung mit Vorhersagen
sns.regplot(x=y_train.iloc[:, 0], y=y_train_pred[:, 0])
plt.show()

#%% Vorhersagen für die (unbekannten) Testdaten erstellen
y_test_pred = model.predict(X_test_scaled)  # (Zeilenanz, 1)  
df_result = y_test
df_result['medv_pred'] = y_test_pred
df_result

ggplot(data=df_result) + aes(x="medv", y="medv_pred") + geom_point() + geom_smooth(method="lm")



#%% plot all variables vs. medv and medv_pred with ggplot
# from plotnine import ggplot, aes, geom_point, geom_smooth, xlim, ggtitle

# # Create and show each plot in a separate matplotlib figure
# for col in boston.columns:
#     if col != 'medv' and col != 'medv_pred':
#         p = (
#             ggplot(boston) +
#             aes(x=col, y="medv") +
#             geom_point() +
#             geom_point(aes(y='medv_pred'), color="red") +
#             geom_smooth(method="lm", se=False) + 
#             xlim(0, 50) +
#             ggtitle(f"{col} vs medv")
#         )
#         # Create a new figure for each plot and display it
#         # Save each plot with a unique filename based on the column name
#         p.save(f"plot_{col}_vs_medv.png")
 

#%% Bestimmtheitsmaß R^2
r2 = r2_score(y_true=y_test['medv'], y_pred=y_test_pred)
print("R^2:", r2)

#%% Exkurs: Dummy Encoding
my_dict = {
    'name': ['Bob', 'Stuart', 'Kevin'],
    'farbe': ['green', 'red', 'blue']
}
minions = pd.DataFrame(my_dict)
# Create dummy encoding for the "farbe" column
farbe_dummies = pd.get_dummies(minions, columns=["farbe"], prefix='farbe', dtype=float)
farbe_dummies

#%% Exkurs: Dreiteilung der Daten in Train/Val/Test
# 1. Aufteilung in Train/Val und Test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# 2. Aufteilung in Train und Val
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=2)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

#%% Resampling Techniques
model = LinearRegression()
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
cross_val_score(estimator=model, X=X, y=y, cv=5, scoring='r2')
# %%
kf = KFold(n_splits=10, shuffle=True, random_state=123)
kf.get_n_splits(X)

#%%
r2_score = []
for train_ind, val_ind in kf.split(X):
    X_train, X_val, y_train, y_val = X[train_ind], X[val_ind], y[train_ind], y[val_ind]
    # model
    # predict
    # r2 score ermitteln
    # r2_score.append(...)
