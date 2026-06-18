#%% Pakete
import pandas as pd

#%% Data Import
file_path = "factbook.csv"
df_factbook = pd.read_csv(file_path, sep=";", skiprows=[1])
df_factbook
#%% Vertrautmachen (Exploratory Data Analysis)
df_factbook.shape  # Anzahl Zeilen und Spalten

#%% numerische Spalten besser verstehen
df_factbook.describe()

#%% Spalten
df_factbook.columns

# %% Filtern
# Ländern mit mindestens 1 Mio. Einwohnern
filter_population = df_factbook['Population'] > 300E6
filter_railways = df_factbook['Railways(km)'] > 100000 # mindestens 10.000 km Eisenbahnlänge hat das Land


df_factbook_filt_pop = df_factbook[filter_population &filter_railways]
df_factbook_filt_pop.shape

#%% Spalten filtern
selected_columns = ['Population', 'Railways(km)']
df_factbook_filt_pop[selected_columns]

# %%
# df_factbook[(df_factbook['Population'] > 1E6) & (df_factbook['Railways(km)'] > 10000)]
# %%
df_factbook.columns

#%% Zeilen nach Indexposition filtern
df_factbook[:3]
# %% Filter von Indexposition und Spaltenname
df_factbook.loc[:3, selected_columns]

#%% rein Spalten und Zeilen nach Index filtern
df_factbook.iloc[:3, :3]

#%% neue Spalten erstellen
df_factbook["Population_million"] = df_factbook["Population"] / 1E6
# %%
