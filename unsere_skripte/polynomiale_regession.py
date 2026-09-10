#%% packages
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_line
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import random
# %% Data Preparation
sample_data = pd.DataFrame(np.arange(-20,40, 0.5), columns=['x'])
sample_data['y'] = 50 + 0.25 * (sample_data['x']-5)**3
sample_data['y_noise'] = sample_data['y'] + np.random.normal(100, 500, sample_data.shape[0])

# %%
(ggplot(sample_data)
 + aes(x = 'x', y = 'y_noise')
 + geom_point()
 + geom_line(aes(y ='y'), color ='red')
)

#%% Model
X_train = sample_data[['x']] # np.array(sample_data['x']).reshape(-1, 1)
y_train = sample_data[['y_noise']] # alternativ np.array(sample_data['y_noise']).reshape(-1, 1)

# prepare the features
degree = 3
poly_feat = PolynomialFeatures(degree=degree)
x_poly = poly_feat.fit_transform(X_train)

# fit the model
model = LinearRegression()
model.fit(x_poly, y_train)

# create predictions
sample_data['y_poly_pred'] = model.predict(x_poly)

# visualise results
(ggplot(sample_data)
 + aes(x = 'x', y = 'y_noise')
 + geom_point()
 + geom_line(aes(y ='y'), color ='red')
 + geom_line(aes(y ='y_poly_pred'), color='green')
)
# %% Calculate R**2 score
r2 = r2_score(sample_data['y_noise'], sample_data['y_poly_pred'])
r2

# %% Adjusted R**2
# calculate adjusted R2 which is better suitable (cost for too many parameters)

# $R^2=1-(1-R^2)*\frac{n-1}{n-p-1}$

p = degree  # nr of independent variables
n = sample_data.shape[0]  # nr of observations
adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
adj_r2

# %%
