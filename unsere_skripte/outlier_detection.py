#%% Pakete
import pandas as pd
import numpy as np
import seaborn as sns
from plotnine import (
    ggplot, aes, geom_histogram, geom_point, geom_smooth,
    geom_boxplot, geom_abline, labs, theme_minimal, facet_wrap,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#%% Datenimport
housing = pd.read_csv("BostonHousing.csv")

#%% Visualisiert eine Spalte und die Verteilung der Punkte innerhalb der Spalte
ggplot(housing) + aes(x='medv') + geom_histogram()


#%% visualisiert die Spalte mittels Boxplot
ggplot(housing) + aes(y='medv') + geom_boxplot()
# %% calculate Q1 and Q3
Q1 = housing['medv'].quantile(0.25)
# Q1 = np.quantile(housing['medv'], q=0.25)
Q3 = housing['medv'].quantile(0.75)

# %% calculate IQR
IQR = Q3 - Q1
# %% calculate lower and upper bound
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#%% Ausreißer im Datensatz kennzeichnen
# (0...kein Ausreißer, 1...Ausreißer)
housing['outlier'] = 0
for val in housing['medv']:
    housing.loc[housing['medv'] == val, 'outlier'] = 1


#%% Alternative Methode
housing['outlier'] = [(1 if val < lower_bound or val > upper_bound else 0) for val in housing['medv']]

# %%
