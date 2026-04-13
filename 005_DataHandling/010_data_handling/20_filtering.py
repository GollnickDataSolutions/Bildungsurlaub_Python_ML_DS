
#%% packages
import pandas as pd

#%%
# source: https://perso.telecom-paristech.fr/eagan/class/igr204/datasets
file_path = 'factbook.csv'
df = pd.read_csv(file_path, sep=';', skiprows=[1])

df
# %% Data Filtering
# select columns
# select columns like df['Country]
type(df['Country'])  # returns Series
type(df[['Country']])  # returns DataFrame
# %%
# 1. find number of unique countries
len(pd.unique(df['Country']))
# %%
# select rows
# 2. get all countries with more than 1E6 people
df[df['Population']>1E6]
# %%
df['Population']>1E6
# %%

df[['Country', 'Population']].shape


# %%
(df['Population']>1E6).value_counts()
# %%
df[df['Population']>1E6]
# %% loc
df.loc[1:3, ['Country']]
# %% iloc
df.iloc[10:20, -4:]  # row 10 to 19, last four cols

# integer location
df.iloc[0,0]  # get the first 

# %%
df.iloc[1, :]  # 2nd row

# %%
df.iloc[:, -1]  # last column

# %% perform multiplication
df['Population'] /1000 * df['Birth rate(births/1000 population)']
# %% find the largest country in the world
df['Population'].sort_values(ascending=False)
# %%
df.loc[[49], :]  # manual approach
# %%
df.loc[[df['Population'].idxmax()], :]
# %%
