#%% Pakete
import pandas as pd
import numpy as np

df = pd.DataFrame({'language': ['R', 'Python', 'SQL', 'R', 'R', 'Python', 'Python'], 
                    'year': [2020, 2020, 2020, 2021, 2022, 2022, 2022],
                    'users': [1E6, 2E6, 0.5E6, 1.1E6, 1.2E6, 2.2E6, 2.4E6]})
df
# %% Aggreation für language über alle anderen Spalten
df.groupby('language').agg(np.mean)

# %% Gruppierung für language , dann Mittelwert für Users ermitteln
df.groupby('language')['users'].mean()
# %%
df.groupby('language').agg({'year': ['count'], 'users': ['mean', 'min', 'max', 'sum']})

#%% Übung: Coffeeshop
df = pd.DataFrame({
    'shop': ['Berlin', 'Hamburg', 'München', 'Berlin', 'Berlin',
             'Hamburg', 'München', 'München', 'Hamburg', 'Berlin'],
    'month': ['Jan', 'Jan', 'Jan', 'Feb', 'Mär',
              'Feb', 'Feb', 'Mär', 'Mär', 'Apr'],
    'cups_sold': [1200, 950, 800, 1300, 1250,
                  1000, 850, 900, 1100, 1400]
})
df

#%% 1. Gesamtanzahl verkaufter cups je Filiale
df.groupby('shop')['cups_sold'].sum()

#%% 2. als Balken- oder Liniendiagramm die Entwicklung darstellen (mit ggplot)
import plotnine as p
df['month'] = pd.Categorical(df['month'], categories=['Jan', 'Feb', 'Mär', 'Apr'], ordered=True)
 
g = (
    p.ggplot(df)
    + p.aes(x='month', y='cups_sold', color='shop', group='shop')  # 'group' sorgt für getrennte Linien je Filiale
    + p.geom_line(size=2)
    # + p.geom_point(size=4)
    + p.labs(x='Monat', y='Verkaufte Tassen', title='Entwicklung der verkauften Tassen je Filiale')
    + p.theme_minimal()
)
g

# %%
