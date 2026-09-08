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

#%% Ausreißer entfernen


starwars = starwars[starwars['mass'] < 1000]
starwars.shape

#%% TODO: IQR - Filter implementieren (Aufgabe Mittwoch früh)

#%% Visualisierung von mass und height
# x = 'height', y = 'mass'
(
    p.ggplot(starwars) 
    + p.aes(x = 'height', y = 'mass')
    + p.geom_point()
    + p.geom_smooth(method='lm', se=False)
    + p.coord_cartesian(ylim=[0, 160])
)

# %% X...beschreibende Merkmale, y...beschriebenes Merkmal (Zielgröße)
X = np.array(starwars["height"]).reshape(-1, 1)
y = np.array(starwars["mass"]).reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)

#%%
