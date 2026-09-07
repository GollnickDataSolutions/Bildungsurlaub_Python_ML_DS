
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