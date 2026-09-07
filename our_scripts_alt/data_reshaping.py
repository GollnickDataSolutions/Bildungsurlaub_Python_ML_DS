#%% Pakete
import pandas as pd
from plotnine import ggplot, aes, geom_tile, facet_wrap

#%%
data = {
    'student': ['Stuart', 'Bob', 'Kevin'],
    'math': [2,3,3],
    'sport': [3,1,2],
    'art': [4,2,1]   
}

#%% Dataframe erstellen df_students_wide
df_students_wide = pd.DataFrame(data)
df_students_wide
# %% wide --> long
df_student_long = df_students_wide.melt(id_vars="student", value_name="noten", var_name="schulfach")
df_student_long
#%% long --> wide
df_student_long.pivot(index='student', columns='schulfach', values='noten').reset_index()#.rename_axis(None, axis=1)
# %%
# 1. Speichern der Daten als csv
# 2. Import der daten in ein Dataframe
# 3. wide --> long Umwandlung
# 4. long --> wide Umwandlung

#%% 2. Import der Daten in ein Dataframe
quartalszahlen = pd.read_csv("quartalszahlen.csv")
quartalszahlen

#%% 3. wide --> long Umwandlung
df_quartalszahlen_long = quartalszahlen.melt(
    id_vars=["Produkt", "Region"],
    value_vars=["Q1_2024", "Q2_2024", "Q3_2024", "Q4_2024"],
    var_name="Quartal",
    value_name="Umsatz",
)
df_quartalszahlen_long

#%% 4. long --> wide Umwandlung
df_quartalszahlen_wide_again = df_quartalszahlen_long.pivot(
    index=["Produkt", "Region"],
    columns=["Quartal"],
    values="Umsatz",
).reset_index().rename_axis(None, axis=1)
df_quartalszahlen_wide_again

# %%  Visualisierung der Quartalszahlen mittels ggplot
(
    ggplot(data=df_quartalszahlen_long) +
    aes(x="Quartal", y="Region", fill="Umsatz") +
    facet_wrap("Produkt", ncol=1) +
    geom_tile() 
)

# %%
