#%% Paket
import pandas as pd
# %% Dataframe
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}
df = pd.DataFrame(data=data)

#%% gib nur Spalte A zurück
df["A"]

#%% alternative (wenn auch unüblich)
df.iloc[:, :1]

# %% Erstelle eine weitere Spalte "D" mit den Werten 10, 11, 12
df["D"] = [10, 11, 12]
df


#%% gib nur die ersten beiden Zeilen zurück
df[:2]

#%% Löschen von Spalte C
df.drop(columns=["C"], inplace=True)
df
# %%
