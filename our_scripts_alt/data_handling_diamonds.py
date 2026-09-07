#%% Pakete laden
import pandas as pd
#%% Daten importieren
df_diamonds = pd.read_csv("Diamonds.csv")
df_diamonds
#%% Vertraut machen mit den Daten
df_diamonds.shape # Anzahl Zeilen und Spalten des Dataframes

#%% Datentypen
df_diamonds.info()

#%% statistische Merkmale der numerischen Spalten
df_diamonds.describe()


#%% Filtern nach "cut", behalte nur Ideal und Premium
df_diamonds["cut"].unique()
filter_best_quality = df_diamonds["cut"].isin(['Ideal', 'Premium'])
df_diamonds_best_quality = df_diamonds[filter_best_quality]

# %% 
from plotnine import ggplot, aes, geom_point, geom_bar, facet_grid, geom_jitter, geom_area, geom_density, labs

# %% Visualisierung einer einzigen Spalte (kategorisch)
ggplot(data=df_diamonds) + aes(x='cut') + geom_bar()

#%% Visualisierung einer einzigen Spalte (numerisch)
ggplot(data=df_diamonds) + aes(x='price') + geom_density()

# %% 2 Variablen:  X kontinuierlich, Y kontinuierlich
(ggplot(data=df_diamonds) + 
  aes(x='x', y='y', color='price', size='carat') + 
  geom_point() + 
  facet_grid("clarity ~ color") +
  labs(x='Länge [cm]', y='Breite [cm]', title = 'Diamanten und ihre Eigenschaften', subtitle='Auswertung des diamonds Datensatzes')
)

