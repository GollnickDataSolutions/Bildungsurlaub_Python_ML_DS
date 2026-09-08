#%% Pakete
import pandas as pd

# %%
data = {
    'student': ['Stuart', 'Bob', 'Kevin'],
    'math': [2,3,3],
    'sport': [3,1,2],
    'art': [4,2,1]
    
}
df_wide = pd.DataFrame(data=data)
df_wide
# %% Breites Format --> Long Format
df_long = df_wide.melt(
    id_vars=['student'],
    var_name='subject',
    value_name='grade'
)

# %% Long Format --> wide format
df_pivot = df_long.pivot(index='student', columns='subject', values='grade').reset_index()
print(df_pivot.columns)  # Index(['student', 'art', 'math', 'sport'], dtype='object', name='subject')

# Mit .rename_axis(None, axis=1) wird der Spaltenname "subject" entfernt:
df_pivot = df_long.pivot(index='student', columns='subject', values='grade').reset_index().rename_axis(None, axis=1)
print(df_pivot.columns)  # Index(['student', 'art', 'math', 'sport'], dtype='object')

# %% Übung: Angestellte
data = {
    'employee': ['Anna', 'Tom', 'Lisa', 'Max'],
    'sales': [5, 3, 4, 2],
    'support': [2, 4, 3, 5],
    'marketing': [3, 3, 5, 1]
}
df_wide = pd.DataFrame(data=data)
df_wide

# %% 1. wide --> long

#%% 2. long --> wide
