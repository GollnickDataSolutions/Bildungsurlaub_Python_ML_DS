#%% Pakete
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import shap
from plotnine import ggplot, geom_col, aes
#%% Elena Import aus csv
# housing = pd.read_csv("housing.csv")
# housing.dropna(inplace=True)
# X = housing.drop(columns="median_house_value")
# y = housing[["median_house_value"]]
# print(X.shape, y.shape)

# %% Daten importieren, bereits getrennt in unabhängige (X) und abhängige Variable (y)
X, y = fetch_california_housing(as_frame=True, return_X_y=True)
# %% Exkurs
# my_tuple = (1, 2)  # 
# my_list = [1, 2]

#%% statistische Eigenschaften des Dataframes
X.describe()

#%% Korrelationsanalyse der einzelnen Merkmale
# sns.heatmap(X.corr(), annot=True)

#%%
# sns.pairplot(X[:1000])

#%% Aufteilen in training und validation
# --> X_train, X_val, y_train, y_val
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, random_state=42, shuffle=True)
y_train

#%% Daten skalieren
scaler = MinMaxScaler()  # StandardScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

# %% Modelltraining
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

#%% Shapley Wert
explainer = shap.TreeExplainer(model)
#%%
shap_values = explainer(X_test_scaled[:10])

#%% Wasserfall-Diagramm erstellen
shap.plots.waterfall(shap_values[1], show=False)
plt.show()

#%% Beeswarm Plot
shap.plots.beeswarm(shap_values, show=False)
plt.show()

#%% Feature-Importance

# Feature Importance DataFrame
df_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot with plotnine (ggplot style)
ggplot(df_importance) + aes(x='Feature', y='Importance') + geom_col()

#%% Testen des Modells
y_test_pred = model.predict(X_test_scaled)
y_test_pred[1]
#%%
sns.regplot(x= y_test, y= y_test_pred)
plt.show()
#%% Bestimmtheitsmaß (R^2) ermitteln
r2_score(y_true=y_test, y_pred=y_test_pred)
