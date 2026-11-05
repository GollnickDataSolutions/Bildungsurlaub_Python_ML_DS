#%% Pakete
import pandas as pd
import os
os.getcwd()
#%% Datenimport
diamonds = pd.read_csv("../001_Datasets/Diamonds.csv")


#%% mit den Daten vertraut machen (EDA...exploratory data analysis)
diamonds.shape  # Anzahl der Zeilen / Spalten
# %% erste bzw. letzte Zeilen des Dataframes anzeigen
print(diamonds.head(2))
print(diamonds.tail(2))

# %% Datentypen
diamonds.info()

#%% Statistische Eigenschaften der numerischen Spalten darstellen
diamonds.describe()
# %% Welche Spalten sind vorhanden
diamonds.columns

#%% Anzahl der eindeutigen Ausprägungen der Spalten
diamonds.nunique()

#%% Eindeutige Ausprägungen
diamonds['cut'].unique()

#%% Gibt es fehlende Wertein den Spalten
diamonds.isna().sum()

#%% Filtern
filter_price = diamonds['price'] < 1000
# filter_cut = diamonds['cut'] == "Good"  # Filter nach genau einem Ausprägungswert
filter_cut = diamonds['cut'].isin(['Ideal', 'Premium', 'Good', 'Very Good']) # Vergleich mit verschiedenen Werten einer Liste
diamonds_filt = diamonds[filter_price & filter_cut]

#%% Spaltenweise filtern (Positivliste definieren)
selected_columns = ['cut', 'price']
diamonds[selected_columns]


# %% Spaltenweise filtern (nur festgelegte Spalten loswerden)
diamonds.drop(columns=selected_columns, inplace=True)
diamonds

# %% Filter nach Indexposition
# innerhalb der eckigen Klammern werden zuerst die Zeilen, dann die Spalten definiert
diamonds.iloc[5:10, -3:]

#%%
diamonds.loc[:10, 'x'].plot()