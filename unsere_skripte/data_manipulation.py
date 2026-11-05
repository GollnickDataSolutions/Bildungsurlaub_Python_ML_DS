#%% Pakete
import pandas as pd
import numpy as np

#%% Data Aggregation
#-------------------
# %% erstellt ein dataframe auf Basis dieses Dictionaries
data = {
    "region":   ["Nord", "Süd", "Nord", "West", "Süd", "West", "Nord"],
    "produkt":  ["A", "B", "A", "C", "A", "B", "C"],
    "umsatz":   [120, 200, 150, 90, 180, 210, 75],
    "menge":    [3, 5, 4, 2, 6, 7, 1],
}

df_sales = pd.DataFrame(data=data)
df_sales

#%% Wieviel Umsatz haben die einzelnen Regionen gebracht?
# df_sales.groupby("region").agg(np.sum)
df_sales.groupby("region").agg("sum")

#%% Wie hoch ist der aggregierte und mittlere Umsatz je Region?
df_sales.groupby("region").agg(["sum", "mean"])

#%% Zusatzaufgabe: Wie hoch ist der maximale und minimale Umsatz je Produkt?
df_sales.groupby("produkt")["umsatz"].agg(["min", "max"])

#%% Data Reshaping (wide and long-format)
#----------------------------------------
student_data = {
    "student": ['Stuart', 'Bob', 'Kevin'],
    "math": [2, 3, 3],
    "sports": [3, 1, 2],
    "art": [4, 2, 1]
}
df_students_wide = pd.DataFrame(student_data)
df_students_wide

#%% breit zu lang (wide to tidy)
df_students_long = df_students_wide.melt(id_vars="student")#, var_name="subject", value_name="grades")

#%% lang zu breit
df_students_long.pivot(index="student", columns="subject", values="grades").reset_index()

# %% Übung: Angestellte
data = {
    'employee': ['Anna', 'Tom', 'Lisa', 'Max'],
    'sales': [5, 3, 4, 2],
    'support': [2, 4, 3, 5],
    'marketing': [3, 3, 5, 1]
}

# %% 1. wide --> long

#%% 2. long --> wide


#%% Data Joining
minions = pd.DataFrame({
    'student': ['Stuart', 'Bob', 'Kevin', 'Gru'],
    'art': [4,2,1, 2]
    
})
print(f"minions:\n {minions}")

despicable_me = pd.DataFrame({
    'student': ['Agnes', 'Margo', 'Edith', 'Gru'],
    'sport': [1,2,2, 3]
    
})
despicable_me
# %% Left Join
minions.merge(despicable_me, on='student', how="left")

# %% Right Join
minions.merge(despicable_me, on='student', how="right")
#%% Inner Join
minions.merge(despicable_me, on='student', how="inner")
#%% Outer Join
minions.merge(despicable_me, on='student', how="outer")

#%% Left Join für minions und frozen mit pd.merge
frozen = pd.DataFrame({
    'character': ['Anna', 'Elsa', 'Olaf'],
    'art': [4,2,1]
})
pd.merge(left=minions, right=frozen, left_on='student', right_on='character', how="outer")
#%%
minions.merge(frozen, left_on='student', right_on='character', how="outer", indicator=True)

#%% 
frozen2 = pd.DataFrame({
    'student': ['Anna', 'Elsa', 'Olaf'],
    'art': [4,2,1]
})

# %% zwei Dataframes übereinander stapeln (vertikal zusammenfügen)
pd.concat([minions, frozen2])

# %%
