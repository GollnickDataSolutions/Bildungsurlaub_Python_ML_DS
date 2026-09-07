#%% Pakete
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#%% Elena Import aus csv
# housing = pd.read_csv("housing.csv")
# housing.dropna(inplace=True)
# X = housing.drop(columns="median_house_value")
# y = housing[["median_house_value"]]
# print(X.shape, y.shape)

# %% Daten importieren, bereits getrennt in unabhängige (X) und abhängige Variable (y)
X, y = fetch_california_housing(as_frame=True, return_X_y=True)
# %% Exkurs
my_tuple = (1, 2)  # 
my_list = [1, 2]

#%% Korrelationsanalyse der einzelnen Merkmale
sns.heatmap(X.corr(), annot=True)

#%%
sns.pairplot(X[:1000])

#%% Aufteilen in training und validation
# --> X_train, X_val, y_train, y_val
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, random_state=42, shuffle=True)
y_train
# %% Modelltraining
model = LinearRegression()
model.fit(X_train, y_train)
model.coef_

#%% Testen des Modells
y_test_pred = model.predict(X_test)

#%%
sns.regplot(x= y_test, y= y_test_pred)
plt.show()
#%% Bestimmtheitsmaß (R^2) ermitteln
r2_score(y_true=y_test, y_pred=y_test_pred)
