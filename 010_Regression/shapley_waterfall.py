#%%
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Daten laden
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modell trainieren
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

#%%
# SHAP-Werte berechnen (TreeExplainer ist effizient für Baum-Modelle)
explainer = shap.TreeExplainer(model)

# Auf einer kleinen Stichprobe berechnen, um Rechenzeit zu sparen
X_sample = X_test.iloc[:100]
shap_values = explainer(X_sample)

#%%
# Waterfall-Plot für eine einzelne Vorhersage (erste Beobachtung)
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
plt.show()

#%%
# Beeswarm-Plot: Überblick über die wichtigsten Merkmale (alle Beobachtungen)
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.show()

