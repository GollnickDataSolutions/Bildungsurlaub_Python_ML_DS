#%% Pakete
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_smooth, geom_text, xlim, ylim, geom_line
from sklearn.linear_model import LinearRegression

# %% Datenimport
starwars = pd.read_csv("starwars.csv")
starwars.head(2)

starwars.shape

#%% NA-Filter
starwars.dropna(inplace=True)
starwars.shape

#%% Filter nach Masse <1000
filter_mass = starwars["mass"] < 1000
starwars_filt = starwars[filter_mass]
starwars_filt.shape

#%% Visualisierung y...mass, x...height, label...name, geom_text()
g = (ggplot(starwars_filt)
 + aes(x='height', y = 'mass', label = 'name') 
 + geom_point()
 + geom_smooth(method = 'lm', se=False, color="red") # Regressionsgerade
 + geom_text()
 + xlim([0, 200])
 + ylim([-50, 200])
)
g

#%% separiere beschreibende Merkmale (X) von abhängigem Merkmal (y)
# X = starwars_filt['height']  # shape: (58, ) <-- Zeilenvektor
X = starwars_filt[['height']]  # shape: (58, 1) 
y = starwars_filt[['mass']]

# %% Modellierung
model = LinearRegression()
model.fit(X, y)

# %% Modellparameter
model.coef_  # Steigungswert (Gewichtszunahme pro cm Größenzuwachs)
# model.intercept_  # Schnittpunkt mit der y-Achse

#%% Vorhersagen erstellen
starwars_filt["mass_predicted"] = model.predict(X)

#%%
g = (ggplot(starwars_filt)
 + aes(x='height', y = 'mass', label = 'name') 
 + geom_point()
 + geom_smooth(method = 'lm', se=False, color="red") # Regressionsgerade
 + geom_line(aes(y = 'mass_predicted'))
 + geom_text()
 + xlim([50, 250])
 + ylim([0, 200])
)
g
# %%
