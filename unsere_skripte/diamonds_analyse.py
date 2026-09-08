
#%% Pakete laden
import pandas as pd
import os


#%% Laden des Dataframes
# Datei befindet sich relativ unter "..\\001_Datasets\\Diamonds.csv"
# Linux: "../001_Datasets/Diamonds.csv"
diamonds_file_path = os.path.join("..", "001_Datasets", "Diamonds.csv")
df_diamonds = pd.read_csv(diamonds_file_path)
df_diamonds
# %% Exploratory Data Analysis (EDA) 
# 
# mit dem Dataframe vertraut machen
df_diamonds.describe()

#%%
df_diamonds.head(2)  # ersten 2 Zeilen 
df_diamonds.tail(2)  # letzte 2 Zeilen

#%% 
df_diamonds.info()

#%% Spalten umbenennen
df_diamonds.rename(columns={'carat': 'karat'}, inplace=True)
  
#%% Spalten löschen
df_diamonds.drop(columns=['color', "clarity"], inplace=True)
df_diamonds

#%% Spaltennamen extrahieren
df_diamonds.columns
# %% Filtern nach Indexposition
# liefere die ersten 5 Zeilen (0, 1, 2, 3, 4) und letzten 4 Spalten
df_diamonds.iloc[:5, -4:]

#%% Filter Zeile 3-8 und behalte alle Spalte
df_diamonds.iloc[3:9, :]


# %% Zeilenfilter zunächst definieren und anschließend anwenden
filter_for_cut = df_diamonds["cut"].isin(["Good", "Very Good"])
filter_price = df_diamonds["price"]< 500
# 2. Filtern nach price < 500
# False False
# False True
# True True  <- nur diese werden behalten

df_diamonds.loc[filter_for_cut & filter_price]

#%%
df_diamonds[filter_for_cut]

#%% Visualisierung der Daten
# from plotnine import * # das hier eher nicht verwenden

# from plotnine import ggplot, aes, geom_bar, geom_point, geom_density, geom_histogram, facet_grid, facet_wrap

import plotnine as p

#%% 1 Dimension (diskrete Variable)
g = (
    p.ggplot(data=df_diamonds) 
    + p.aes(x='cut') 
    + p.geom_bar()
) 
g

#%%
from pandas.api.types import CategoricalDtype
cut_order = CategoricalDtype(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], ordered=True)
df_diamonds['cut'] = df_diamonds['cut'].astype(cut_order)

#%% 2 Variablen: x='x', y='y'
# Filter cut nach Ideal und Premium
# Bonus: Reihefolge der Subplots ändern
filter_cut = df_diamonds['cut'].isin(['Ideal', 'Premium'])
g = (
    p.ggplot(data=df_diamonds[filter_cut]) 
    + p.aes(x='x', y='y', color='price') 
    + p.geom_point()
    + p.facet_grid(rows='cut', cols='color')
    + p.labs(x='Horizontale Ausdehnung [mm]', y= 'Vertikale Ausdehnung [mm]', title='Diamanten-Analyse', subtitle='Punkteplot für Preis und Größe der Diamanten')
    + p.theme_dark()
    + p.scale_color_gradient()
) 
g
# g.save("diamonds.png")
# %% x=clarity, y=price --> findet eine geeignete Visualisierung
g = (
    p.ggplot(data=df_diamonds[filter_cut])
    + p.aes(x='x', y='y', color='price')
    + p.geom_point()
    + p.facet_grid(rows='cut', cols='color')
    + p.labs(x='Horizontale Ausdehnung [mm]', y='Vertikale Ausdehnung [mm]', title='Diamanten-Analyse', subtitle='Punkteplot für Preis und Größe der Diamanten')
    + p.theme_dark()
    + p.scale_color_gradient()
)
g

#%% Seaborn
import seaborn.objects as so
(
    so.Plot(df_diamonds[filter_cut], x='x', y='y', color='price')
    .facet(row='cut', col='color')
    .add(so.Dot())
)


# %%
import seaborn as sns
penguins = sns.load_dataset("penguins").dropna()
penguins.to_csv("penguins.csv")