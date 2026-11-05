#%% Pakete
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_errorbar, labs, theme_minimal, geom_smooth

#%% Daten importieren
hubble = pd.read_csv('Hubble.csv')
hubble.head()

#%% Messdaten darstellen
(ggplot(hubble)
 + aes(y='v', x='D')
 + geom_point(color='black')
 + labs(y='Geschwindigkeit v [km/s]', x='Entfernung D [MPc]', title='Hubble-Messdaten')
 + theme_minimal()
 + geom_smooth(method="lm")
)

# %% Trennung von unabh. (X) und abh. Variable (y)
X = hubble[["D"]]
y = hubble[["v"]]
#%% Modellinstanz erstellen und trainieren
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

#%% Modellparameter anzeigen
model.coef_
# model.intercept_

#%%

