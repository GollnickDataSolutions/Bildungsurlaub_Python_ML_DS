#%% Pakete
import folium as f
import pandas as pd
import numpy as np

# %% Daten vorbereiten
german_cities = {'city': ['Hamburg', 'Berlin', 'Munich'],
                  'lat': [53.5511, 52.5200, 48.1351],
                  'long': [9.9937, 13.4050, 11.5820]
}
german_cities

#%% 1. Erzeugung eines Dataframes
df_cities = pd.DataFrame(german_cities)
df_cities
#%% Zentraler Punkt der Karte wird ermittelt
# Lösung mit pandas Methoden
center = [df_cities['lat'].mean(), df_cities['long'].mean()]

#%% Lösung mit numpy Methoden
center = [np.mean(df_cities['lat']), np.mean(df_cities['long'])]

# %% Karte erstellen
map = f.Map(location=center, zoom_start=4)

# %% jede Stadt des Dataframes als Punkt darstellen
# 1. Schleife, die über alle Zeilen des df_ läuft

# 2. Im Schleifenbody wird f.CircleMarker ausgeführt
#    dabei ist die location df_cities.loc[i, 'lat'], df_cities.loc[i, 'long']
# f.CircleMarker(location=[])
for i in range(len(df_cities)):
    print(i)
    f.CircleMarker(location=[df_cities.loc[i, 'lat'], df_cities.loc[i, 'long']]).add_to(map)


# %%
map

# %%
