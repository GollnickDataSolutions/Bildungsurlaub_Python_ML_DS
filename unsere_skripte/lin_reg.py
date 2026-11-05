#%% Pakete
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_smooth, labs, geom_text, xlim, ylim

#%% Starwars Daten importieren
starwars = pd.read_csv("Starwars.csv")
starwars.head()
starwars.shape

#%% Filterung nach Masse < 1000
filter_mass = starwars["mass"] < 1000
starwars_filt = starwars[filter_mass]
starwars_filt.shape
#%% Visualisierung erstellen
(ggplot(starwars_filt)
 + aes(x="height", y="mass", label="name")
 + geom_point()
 + geom_text()
 + geom_smooth(method="lm", se=False)
 + labs(title="Größe und Masse", x="Größe (cm)", y="Masse (kg)")
 + xlim(0, 300)
 + ylim(-50, 200)
)

# %% 
starwars_filt
# %% Trennung von unabh. und abh. Variablen; 
# X (unabhängige Merkmale), y (abhängiges Merkmal)
X = starwars_filt[["height"]]
y = starwars_filt[["mass"]]

#%% Lineares Regressionsmodell trainieren
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

#%% Modellparameter anzeigen
model.coef_
model.intercept_
# %%
starwars_filt["mass_pred"] = model.predict(X)
(ggplot(starwars_filt)
 + aes(x="height", y="mass", label="name")
 + geom_point()
 + geom_point(aes(y="mass_pred"), color="red")
#  + geom_text()
 + geom_smooth(method="lm", se=False, color="blue")
 + labs(title="Größe und Masse", x="Größe (cm)", y="Masse (kg)")
)

#%% Gütemaße, R^2 (Coefficient of Determination)
# gibt an, wieviel Streuung der Daten durch das Modell erklärt wird
from sklearn.metrics import r2_score
r2_score(y_true=starwars_filt["mass"], y_pred=starwars_filt["mass_pred"])

#%%
