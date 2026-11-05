#%% Pakete
import pandas as pd
# from plotnine import ggplot
import plotnine as pl
#%% Diamonds Data Importieren
diamonds = pd.read_csv("../001_Datasets/Diamonds.csv")
diamonds.shape

#%% 1-Variable (kategorisch)
pl.ggplot(data=diamonds) + pl.aes(x='cut', fill='color') + pl.geom_bar()

#%% 1-variable price (kontinuierlich)
(pl.ggplot(diamonds) +
  pl.aes(x = 'price') + 
  pl.geom_density()
)

#%% 2 variablen (x...cont, y...cont)
(
    pl.ggplot(diamonds) + 
    pl.aes(x='x', y='y', color='price', size='carat') + 
    pl.geom_point() +
    pl.facet_grid('color ~cut') +
    pl.labs(x='Horizontale Ausdehnung [mm]', y='Vertikale Ausdehnung [mm]', title='Diamanten', subtitle='Detaillierte Analyse') +
    pl.theme_538()
)
# %%
