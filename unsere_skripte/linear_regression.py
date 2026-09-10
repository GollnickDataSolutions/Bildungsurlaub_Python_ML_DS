#%% Pakete

import plotnine as p
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
#%% Daten laden
file_path = "../001_Datasets/Starwars.csv"
starwars = pd.read_csv(file_path)
starwars.dropna(inplace=True)
# %% anzahl zeilen und spalten ermitteln
starwars.shape

#%% Ausreißer entfernen (manueller Ansatz)
starwars = starwars[starwars['mass'] < 1000]
starwars.shape

#%% IQR - Filter implementieren (Interquartilsabstand-Filter für "mass" und "height")
Q1 = starwars['mass'].quantile(0.25)
Q3 = starwars['mass'].quantile(0.75)
IQR = Q3 - Q1
print(f"Unsere IQR beträgt {IQR:.1f} kg")
IQR_FACTOR = 3  # 1.5 (milde Ausreißer), 3 (extreme Ausreißer)
lower_bound = Q1 - IQR_FACTOR * IQR
upper_bound = Q3 + IQR_FACTOR * IQR
# Anwendung der Grenzen beim Filtern
starwars_filt = starwars[(starwars['mass']>= lower_bound) & (starwars['mass']<= upper_bound)]

#%%
starwars.shape

#%% Visualisierung von mass und height
# x = 'height', y = 'mass'
(
    p.ggplot(starwars_filt) 
    + p.aes(x = 'height', y = 'mass')
    + p.geom_point()
    + p.geom_smooth(method='lm', se=False)
    + p.coord_cartesian(ylim=[0, 160], xlim=[0, 300])
)

# %% X...beschreibende Merkmale, y...beschriebenes Merkmal (Zielgröße)
X = starwars[["height"]] #np.array(starwars["height"]).reshape(-1, 1)
y = starwars[["mass"]]  #np.array(starwars["mass"]).reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)

#%% Parameter
model.intercept_  # Schnittpunkt mit der y-Achse

#%% 
model.coef_  # Steigungswert (1cm Größenänderung bewirkt x kg Gewichtsänderung)

#%% Vorhersagen ermitteln
starwars['mass_pred'] = model.predict(X)
starwars
#%% Exkurs: Pandas Serie vs. Dataframe
starwars[["height"]]

#%% Visualisierung der Vorhersagen
(
    p.ggplot(starwars) 
    + p.aes(x = 'height', y = 'mass')
    + p.geom_point()
    + p.geom_point(p.aes(y='mass_pred'), color="red")
    + p.geom_smooth(method='lm', se=False)
    + p.coord_cartesian(ylim=[0, 160], xlim=[0, 300])
)

#%% Metrik zum Überprüfen der Modellgüte
r2 = r2_score(y_true=starwars["mass"], y_pred=starwars["mass_pred"])
r2
#%% Exkurs: Strings mit einfachen und doppelten Anführungszeichen
# der Unterschied wird nur relevant z.B. bei Apostroph im String
"Bert's Kurs"
'Bert\'s Kurs'