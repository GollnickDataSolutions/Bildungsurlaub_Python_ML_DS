#%% Pakete
import pandas as pd
import numpy as np
import seaborn as sns
from plotnine import (
    ggplot, aes, geom_histogram, geom_point, geom_smooth,
    geom_boxplot, geom_abline, labs, theme_minimal, facet_wrap,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#%% Datenimport
housing = pd.read_csv("BostonHousing.csv")

#%% EDA: Überblick
print(housing.shape)
print(housing.head())
housing.info()

#%% EDA: Statistik, fehlende Werte, Duplikate
print(housing.describe().T)
print(housing.isna().sum())
print("Duplikate:", housing.duplicated().sum())
print(housing.nunique())

#%% Visualisierung: Verteilung der Zielvariable
(
    ggplot(housing, aes(x="medv"))
    + geom_histogram(bins=30, fill="steelblue", color="white")
    + labs(title="Verteilung des Hauspreises", x="medv (in 1000 USD)", y="Anzahl")
    + theme_minimal()
)
#%% EDA: Korrelation mit Zielvariable medv (Median-Hauspreis in 1000 USD)
corr = housing.corr(numeric_only=True)
print(corr["medv"].sort_values(ascending=False))


#%% Visualisierung: Korrelationsmatrix (Heatmap)
ax = sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
)
ax.set_title("Korrelationsmatrix")
ax.set_xlabel("")
ax.set_ylabel("")

#%% Visualisierung: Zusammenhang der wichtigsten Features mit medv
top_features = ["lstat", "rm", "ptratio", "indus"]
long = housing.melt(id_vars="medv", value_vars=top_features,
                    var_name="feature", value_name="value")
(
    ggplot(long, aes(x="value", y="medv"))
    + geom_point(alpha=0.4, color="steelblue")
    + geom_smooth(method="lm", color="firebrick")
    + facet_wrap("feature", scales="free_x")
    + labs(title="Preis vs. wichtigste Features", x="", y="medv")
    + theme_minimal()
)

#%% Visualisierung: Preis nach Charles-River-Lage
(
    ggplot(housing, aes(x="factor(chas)", y="medv"))
    + geom_boxplot(fill="lightsteelblue")
    + labs(title="Preis nach Lage am Charles River", x="chas", y="medv")
    + theme_minimal()
)

#%% Lineare Regression: Train/Test-Split
X = housing.drop(columns=["medv", "chas"]).astype(float)
y = housing["medv"]
print(X.shape)
print(y.shape)
#%% Train und Test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=42)
print(X_train.shape)
print(X_test.shape)

#%% Skalierung / Standardisierung der Daten
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Modellierung
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(X_train_scaled, y_train)

#%% Parameter überprüfen
model.coef_

#%% Vorhersagen erstellen
y_pred_test = model.predict(X_test_scaled)
y_pred_test
#%% Visualisierung der Vorhersagen gegenüber den echten Werten
df_results = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred_test
})
df_results
# %%
ggplot(df_results) + aes(x='y_true', y='y_pred') + geom_point() + geom_smooth(method="lm")
# %% Evaluierung des R^2
from sklearn.metrics import r2_score
r2_score(y_true=df_results['y_true'], y_pred=df_results['y_pred'])



