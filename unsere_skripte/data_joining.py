#%% Pakete
import pandas as pd
import numpy as np

#%%
minions = pd.DataFrame({
    'student': ['Stuart', 'Bob', 'Kevin', 'Gru'],
    'art': [4,2,1, 2]
    
})
print(f"minions:\n {minions}")

despicable_me = pd.DataFrame({
    'student': ['Agnes', 'Margo', 'Edith', 'Gru'],
    'sport': [1,2,2, 3]
    
})
print(f"despicable me:\n {despicable_me}")
# %% Left-Join
minions.merge(right=despicable_me, how="left", on="student")
# %% Right-Join
minions.merge(right=despicable_me, how="right", on="student")
# %% Inner-Join
minions.merge(right=despicable_me, how="inner", on="student")

#%%
minions_2 = pd.DataFrame({
    'person': ['Stuart', 'Bob', 'Kevin', 'Gru'],
    'art': [4,2,1, 2]
})
minions_2.merge(right=despicable_me, how="right", left_on="person", right_on="student")

#%% Übung: Hunde und Hundehalter
owners = pd.DataFrame({
    'pet': ['Bello', 'Mia', 'Rocky', 'Luna'],
    'age': [3, 5, 2, 4]
})

vet_visits = pd.DataFrame({
    'pet': ['Mia', 'Rocky', 'Whiskers', 'Luna'],
    'last_checkup': ['2023-05-01', '2023-06-15', '2023-04-20', '2023-07-10']
})

# 1. Inner join