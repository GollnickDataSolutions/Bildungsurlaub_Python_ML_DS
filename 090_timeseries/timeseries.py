#%% Packages
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Time series specific imports
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ggplot imports
from plotnine import (ggplot, aes, geom_line, geom_histogram, geom_point, 
                      labs, theme_minimal, theme, 
                      element_text, facet_wrap, scale_color_manual,
                      scale_linetype_manual)


#%% Data Import and Preparation
data = sns.load_dataset("flights")
print(f'Number of Entries: {len(data)}')
print(data.head())
print(f'\nData shape: {data.shape}')
print(f'Data types:\n{data.dtypes}')

# Create datetime index
data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + 
                               data['month'].astype(str) + '-01')
data = data.set_index('date')
data = data.sort_index()

# We'll focus on the 'passengers' column
ts = data['passengers']
print(f'\nTime series range: {ts.index.min()} to {ts.index.max()}')

#%% Exploratory Data Analysis
# Prepare data for plotting
ts_df = ts.reset_index()
ts_df.columns = ['date', 'passengers']

# Original time series plot
p1 = (ggplot(ts_df, aes(x='date', y='passengers'))
      + geom_line(size=1.2)
      + labs(title='Air Passengers Time Series (1949-1960)',
             x='Date',
             y='Number of Passengers')
      + theme_minimal()
      + theme(figure_size=(14, 4),
              plot_title=element_text(size=14, weight='bold'))
)
p1


#%% Time Series Decomposition
# Decompose the time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(ts, model='multiplicative', period=12)

# Prepare decomposition data for plotting
decomp_df = pd.DataFrame({
    'date': ts.index,
    'original': ts.values,
    'trend': decomposition.trend.values,
    'seasonal': decomposition.seasonal.values,
    'residual': decomposition.resid.values
})

# Reshape for faceting
decomp_long = pd.melt(decomp_df, id_vars=['date'], 
                      value_vars=['original', 'trend', 'seasonal', 'residual'],
                      var_name='component', value_name='value')

# Create labels mapping
component_labels = {
    'original': 'Original',
    'trend': 'Trend',
    'seasonal': 'Seasonal',
    'residual': 'Residual'
}
decomp_long['component_label'] = decomp_long['component'].map(component_labels)

# Color mapping
colors = {'original': 'blue', 'trend': 'orange', 'seasonal': 'green', 'residual': 'red'}

# Decomposition plot with facets
p_decomp = (ggplot(decomp_long, aes(x='date', y='value'))
            + geom_line(size=1.2)
            + facet_wrap('~component_label', ncol=1, scales='free_y')
            + labs(title='Time Series Decomposition',
                   x='Date',
                   y='Value')
            + theme_minimal()
            + theme(figure_size=(14, 10),
                    plot_title=element_text(size=14, weight='bold', ha='center'),
                    strip_text=element_text(size=11, weight='bold'))
)
p_decomp

#%% Train-Test Split
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# Prepare data for plotting
train_df = train.reset_index()
train_df.columns = ['date', 'passengers']
train_df['dataset'] = 'Training Data'

test_df = test.reset_index()
test_df.columns = ['date', 'passengers']
test_df['dataset'] = 'Test Data'

split_df = pd.concat([train_df, test_df])

# Visualization
p_split = (ggplot(split_df, aes(x='date', y='passengers', color='dataset'))
           + geom_line(size=1.2)
           + labs(title='Train-Test Split',
                  x='Date',
                  y='Number of Passengers',
                  color='Dataset')
           + scale_color_manual(values=['#1f77b4', '#ff7f0e'])
           + theme_minimal()
           + theme(figure_size=(14, 6),
                   plot_title=element_text(size=14, weight='bold', ha='center'),
                   legend_position='right')
)
p_split

#%% Model 1: ARIMA
print('\n' + '='*60)
print('MODEL 1: ARIMA')
print('='*60)

# Fit ARIMA model (p,d,q) = (1,1,1)
arima_model = ARIMA(train, order=(1, 1, 1))
arima_fit = arima_model.fit()
print(arima_fit.summary())

# Predictions
arima_pred = arima_fit.forecast(steps=len(test))
arima_pred.index = test.index

#%% Model 2: SARIMA (Seasonal ARIMA)
print('\n' + '='*60)
print('MODEL 2: SARIMA')
print('='*60)

# Fit SARIMA model with seasonal component
# (p,d,q) x (P,D,Q,s) where s=12 for monthly data
sarima_model = SARIMAX(train, order=(1, 1, 1), 
                       seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)
print(sarima_fit.summary())

# Predictions
sarima_pred = sarima_fit.forecast(steps=len(test))
sarima_pred.index = test.index


#%% Model 3: Exponential Smoothing (Holt-Winters)
print('\n' + '='*60)
print('MODEL 3: Exponential Smoothing (Holt-Winters)')
print('='*60)

# Fit Holt-Winters model
hw_model = ExponentialSmoothing(train, 
                                seasonal_periods=12,
                                trend='add',
                                seasonal='add')
hw_fit = hw_model.fit()

# Predictions
hw_pred = hw_fit.forecast(steps=len(test))
hw_pred.index = test.index


#%% Model 4: LSTM Neural Network
#%% Model Comparison
print('\n' + '='*60)
print('MODEL COMPARISON')
print('='*60)

# Create comparison dataframe
comparison = pd.DataFrame({
    'Model': ['ARIMA', 'SARIMA', 'Holt-Winters']
})

print('\n', comparison.to_string(index=False))

#%% Prepare data for individual model plots
train_plot = train.reset_index()
train_plot.columns = ['date', 'passengers']
train_plot['type'] = 'Training Data'

test_plot = test.reset_index()
test_plot.columns = ['date', 'passengers']
test_plot['type'] = 'Actual Test Data'

sarima_plot = sarima_pred.reset_index()
sarima_plot.columns = ['date', 'passengers']
sarima_plot['type'] = 'SARIMA Prediction'
sarima_plot['model'] = 'SARIMA'

hw_plot = hw_pred.reset_index()
hw_plot.columns = ['date', 'passengers']
hw_plot['type'] = 'Holt-Winters Prediction'
hw_plot['model'] = 'Holt-Winters'

#%% Create individual plots
models_data = [
    ('SARIMA', sarima_plot),
    ('Holt-Winters', hw_plot)
]

for name, pred_df in models_data:
    plot_df = pd.concat([
        train_plot.assign(model=name),
        test_plot.assign(model=name),
        pred_df
    ])
    
    p_model = (ggplot(plot_df, aes(x='date', y='passengers', color='type', linetype='type'))
               + geom_line(size=1.2)
               + labs(title=f'{name}',
                      x='Date',
                      y='Number of Passengers',
                      color='Type',
                      linetype='Type')
               + scale_color_manual(values=['#1f77b4', 'black', '#ff7f0e'])
               + scale_linetype_manual(values=['solid', 'solid', 'dashed'])
               + theme_minimal()
               + theme(figure_size=(8, 5),
                       plot_title=element_text(size=12, weight='bold', ha='center'),
                       legend_position='right')
    )
    p_model

# Combined comparison plot
combined_df = pd.concat([
    train_plot.assign(model='All Models'),
    test_plot.assign(model='All Models'),
    sarima_plot,
    hw_plot
])

p_combined = (ggplot(combined_df, aes(x='date', y='passengers', color='type', linetype='type'))
              + geom_line(size=1.2, alpha=0.8)
              + labs(title='All Models Comparison',
                     x='Date',
                     y='Number of Passengers',
                     color='Type',
                     linetype='Type')
              + scale_color_manual(values=['#1f77b4', 'black', '#ff7f0e', '#2ca02c', '#d62728'])
              + scale_linetype_manual(values=['solid', 'solid', 'dashed', 'dashed', 'dashed'])
              + theme_minimal()
              + theme(figure_size=(14, 7),
                      plot_title=element_text(size=14, weight='bold', ha='center'),
                      legend_position='right')
)
p_combined



#%% Future Forecasting
print('\n' + '='*60)
print('FUTURE FORECASTING (Next 12 Months)')
print('='*60)

# Retrain models on full dataset for future forecasting
future_steps = 12

# ARIMA
arima_full = ARIMA(ts, order=(1, 1, 1)).fit()
arima_future = arima_full.forecast(steps=future_steps)

# SARIMA
sarima_full = SARIMAX(ts, order=(1, 1, 1), 
                      seasonal_order=(1, 1, 1, 12)).fit(disp=False)
sarima_future = sarima_full.forecast(steps=future_steps)

# Holt-Winters
hw_full = ExponentialSmoothing(ts, seasonal_periods=12,
                               trend='add', seasonal='add').fit()
hw_future = hw_full.forecast(steps=future_steps)

# Create future dates
last_date = ts.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                             periods=future_steps, freq='MS')

# Prepare data for plotting
historical_df = ts.reset_index()
historical_df.columns = ['date', 'passengers']
historical_df['type'] = 'Historical Data'

forecast_list = []
forecast_list.append(pd.DataFrame({
    'date': future_dates,
    'passengers': arima_future.values,
    'type': 'ARIMA Forecast'
}))
forecast_list.append(pd.DataFrame({
    'date': future_dates,
    'passengers': sarima_future.values,
    'type': 'SARIMA Forecast'
}))
forecast_list.append(pd.DataFrame({
    'date': future_dates,
    'passengers': hw_future.values,
    'type': 'Holt-Winters Forecast'
}))

forecast_plot_df = pd.concat([historical_df] + forecast_list, ignore_index=True)

# Visualization
p_forecast = (ggplot(forecast_plot_df, aes(x='date', y='passengers', color='type', linetype='type'))
              + geom_line(size=1.2)
              + geom_point(data=forecast_plot_df[forecast_plot_df['type'] != 'Historical Data'], 
                          size=3, shape='o')
              + labs(title='Future Forecasting (Next 12 Months)',
                     x='Date',
                     y='Number of Passengers',
                     color='Type',
                     linetype='Type')
              + scale_color_manual(values=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
              + scale_linetype_manual(values=['solid', 'dashed', 'dashed', 'dashed'])
              + theme_minimal()
              + theme(figure_size=(14, 7),
                      plot_title=element_text(size=14, weight='bold', ha='center'),
                      legend_position='right')
)
p_forecast
