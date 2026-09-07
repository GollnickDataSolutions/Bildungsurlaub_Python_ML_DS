#%% pakete
import pandas as pd

# %%
minions = pd.DataFrame({
    'student': ['Stuart', 'Bob', 'Kevin', 'Gru'],
    'art': [4,2,1, 2]
    
})
minions

#%%
despicable_me = pd.DataFrame({
    'student2': ['Agnes', 'Margo', 'Edith', 'Gru'],
    'sport': [1,2,2, 3]
    
})
despicable_me
# %% Left join
minions.merge(right=despicable_me, how="left", left_on="student", right_on="student2")
# %% Right join
minions.merge(right=despicable_me, how="right", left_on="student", right_on="student2")

# %% inner join

#%% outer join

#%%
minions2 = pd.DataFrame({
    'student': ['Vector'],
    'art': [4]
    
})
minions2

#%% concat
pd.concat([minions, minions2]).reset_index(drop=True)

#%%

