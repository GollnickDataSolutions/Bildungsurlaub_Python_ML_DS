#%%  Pakete
import seaborn as sns

#%% Daten laden
penguins = sns.load_dataset("penguins")
penguins

#%% Daten als JSON exportieren
# Exportiert den Datensatz nach penguins.json (lesbar formatiert, im
# selben Ordner wie dieses Skript), damit er einfach geteilt werden kann.
from pathlib import Path

json_path = Path(__file__).parent / "penguins.json"
penguins.to_json(json_path, orient="records", indent=2, force_ascii=False)
print(f"JSON gespeichert unter: {json_path} ({len(penguins)} Zeilen)")
# %%
from penguins_data import penguins_data
# %%
import pandas as pd
penguins = pd.DataFrame(penguins_data)


#%%
penguins.to_csv("penguins.csv")


#%%
#%%
# Overall trend: looks negative
(ggplot(penguins.dropna(), aes('bill_length_mm', 'bill_depth_mm'))
+ geom_point()
+ geom_smooth(method='lm', color='black'))

#%%
from plotnine import ggplot, aes, geom_point, geom_smooth, scale_color_brewer, theme_minimal, labs
# Per species: each is positive! (the paradox)
(ggplot(penguins.dropna(), aes('bill_length_mm', 'bill_depth_mm', color='species'))
+ geom_point()
+ geom_smooth(method='lm')
+ scale_color_brewer(type='qual', palette='Set1')
+ theme_minimal()
+ labs(title='Simpsons Paradoxon bei den Pinguinen'))

#%% Data Aggregation
# Hypothesis: Flipper length hängt mit dem Geschlecht zusammen
import numpy as np
penguins[["sex", "bill_length_mm","flipper_length_mm"]].groupby("sex").agg([np.mean])

#%% Filtern nach fehlenden Werten


#%%
penguins.groupby("sex").agg({"bill_length_mm": [np.mean, np.median], "flipper_length_mm": [np.mean]})

# Ü1: Anzahl Pinguine pro Insel

# Ü2: mittleres Gewicht nach Geschlecht
penguins[['sex','body_mass_g']].groupby('sex').agg({'body_mass_g': [np.mean, np.median]})
# Ü3: Anzahl Pinguine pro Insel/Spezies
