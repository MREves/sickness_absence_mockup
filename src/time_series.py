# file containing functions for time series forecasting models

#--------------------
#Import libraries
#--------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import sys
import plotly.graph_objects as go
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Force the project root into the path so imports always work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.formatting import debug_print

#--------------------
#Import other files
#--------------------
from src.data_factory import generate_nhs_dummy_data

#--------------------
#Debugging
#--------------------
DEBUG_MODE = False

#--------------------
#Create functions
#--------------------

def plot_time_series(df, date_col="date", value_col="y", title="Synthetic Time Series Data"):
    """
    Plots a time series given a DataFrame and column names.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df[value_col], label="Time Series Data")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


#--------------------

def plot_forecast_comparison(train_series, test_series, arima_forecast, arima_ci, naive_forecast, naive_ci, future_forecast=None, future_ci=None, title="Forecast Comparison", historical_periods=24):
    """
    Visualizes training data, test data, and two forecasts with confidence intervals using Plotly.
    Shows only the last `historical_periods` of the training data to prevent squashing the forecast.
    """
    fig = go.Figure()
    
    # Plot historical data
    plot_train = train_series.iloc[-historical_periods:] if historical_periods is not None else train_series
    
    fig.add_trace(go.Scatter(x=plot_train.index, y=plot_train, mode='lines', 
                             name='Historical Data', line=dict(color='black')))
                             
    fig.add_trace(go.Scatter(x=test_series.index, y=test_series, mode='lines+markers', 
                             name='Actual Values (Test)', line=dict(color='blue', dash='dash')))
                             
    # Plot ARIMA forecast
    fig.add_trace(go.Scatter(x=test_series.index, y=arima_forecast, mode='lines', 
                             name='SARIMA Forecast', line=dict(color='red')))
                             
    fig.add_trace(go.Scatter(x=list(test_series.index) + list(test_series.index)[::-1],
                             y=list(arima_ci[:, 1]) + list(arima_ci[:, 0])[::-1],
                             fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', 
                             line=dict(color='rgba(255,255,255,0)'),
                             name='SARIMA 95% CI', showlegend=False))
                             
    # Plot Naive forecast
    fig.add_trace(go.Scatter(x=test_series.index, y=naive_forecast, mode='lines', 
                             name='Seasonal Naive Forecast', line=dict(color='green')))
                             
    fig.add_trace(go.Scatter(x=list(test_series.index) + list(test_series.index)[::-1],
                             y=list(naive_ci[:, 1]) + list(naive_ci[:, 0])[::-1],
                             fill='toself', fillcolor='rgba(0, 128, 0, 0.2)', 
                             line=dict(color='rgba(255,255,255,0)'),
                             name='Naive 95% CI', showlegend=False))
                             
    # Plot Future forecast
    if future_forecast is not None:
        fig.add_trace(go.Scatter(x=future_forecast.index, y=future_forecast, mode='lines', 
                                 name='Future Projection (12 Mo)', line=dict(color='purple', dash='dot')))
                                 
        if future_ci is not None:
            fig.add_trace(go.Scatter(x=list(future_forecast.index) + list(future_forecast.index)[::-1],
                                     y=list(future_ci[:, 1]) + list(future_ci[:, 0])[::-1],
                                     fill='toself', fillcolor='rgba(128, 0, 128, 0.2)', 
                                     line=dict(color='rgba(255,255,255,0)'),
                                     name='Future 95% CI', showlegend=False))
                             
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Daily Absences",
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig


#--------------------

class TimeSeriesProcessor:
    def __init__(self, period=12):
        self.period = period
        self.optimal_lambda = None
        self.trend_type = "additive"
        self.shift_value = 0
        self.is_fitted = False

    def clean_data(self, series):
        series = pd.Series(series).copy().astype(float)
        series = series.interpolate(method='linear').ffill().bfill()

        if len(series) > 2 * self.period:
            stl = STL(series, period=self.period, robust=True).fit()
            resid = stl.resid
            # Robust Outlier Detection
            sigma = resid.std()
            outliers = (resid - resid.median()).abs() > (3 * sigma)
            if outliers.any():
                series[outliers] = stl.trend[outliers] + stl.seasonal[outliers]
        return series

    def fit_transform(self, series):
        """Prepare series: Clean and Log-Transform only if multiplicative."""
        series = self.clean_data(series)
        
        # 1. Positivity for Box-Cox
        self.shift_value = abs(series.min()) + 0.01 if (series <= 0).any() else 0
        series_shifted = series + self.shift_value
            
        _, self.optimal_lambda = boxcox(series_shifted)
        # If lambda < 0.5, we use Log to stabilize variance (Multiplicative)
        self.trend_type = "multiplicative" if abs(self.optimal_lambda) < 0.5 else "additive"
        
        transformed = np.log(series_shifted) if self.trend_type == "multiplicative" else series_shifted
        self.is_fitted = True
        return transformed

    def fit_best_model(self, transformed_series):
        """Fits ARIMA on Cleaned/Scaled data. Let ARIMA handle 'd'."""
        return auto_arima(
            transformed_series, 
            seasonal=True, m=self.period,
            stepwise=True, suppress_warnings=True,
            error_action='ignore'
        )

    def inverse_transform(self, forecast):
        """Undo Log/Box-Cox only. No diff inversion needed if ARIMA handled d."""
        if self.trend_type == "multiplicative":
            forecast = np.exp(forecast)
        
        return forecast - self.shift_value
    
    def generate_seasonal_naive_forecast(self, series, n_periods=12):
        """
        Predicts the next n_periods by looking back exactly one full cycle.
        E.g., Next January = This January.
        Also returns confidence intervals based on historical residuals.
        """
        series = pd.Series(series).dropna()
        
        if len(series) < self.period:
            raise ValueError(f"Series too short for seasonal naive (need at least {self.period} points).")
        
        # Take the last 'period' observations
        last_cycle = series.iloc[-self.period:].values
        
        # Repeat that cycle as many times as needed to fill n_periods
        repeats = int(np.ceil(n_periods / self.period))
        forecast = np.tile(last_cycle, repeats)[:n_periods]
        
        # 2. Calculate Confidence Intervals
        # Residuals are the difference between actuals and the value from one season ago
        residuals = series.iloc[self.period:] - series.iloc[:-self.period]
        # Std dev of residuals is our forecast error
        error_std = residuals.std()
        
        # Interval is forecast +/- z * std_error
        # For 95% CI, z is approx 1.96
        conf_int = np.array([
            forecast - 1.96 * error_std,
            forecast + 1.96 * error_std
        ]).T # Transpose to get (n_periods, 2) shape
        
        return forecast, conf_int

def run_time_series_analysis(ts_df_pd=None):
    #--------------------
    #Create data
    #--------------------
    if ts_df_pd is None:
        ts_df, _ = generate_nhs_dummy_data()
        ts_df_pd = ts_df.to_pandas().set_index('date')
    
    # Resample to monthly frequency to match seasonality period
    ts_monthly = ts_df_pd['y'].resample('MS').sum()

    #1. Split for validation
    train = ts_monthly[:-12]
    test = ts_monthly[-12:]

    # 2. SARIMA Path
    processor = TimeSeriesProcessor(period=12)
    stationary_train = processor.fit_transform(train)
    model = processor.fit_best_model(stationary_train)
    # Get forecast and confidence intervals
    arima_raw, arima_conf_int_raw = model.predict(n_periods=12, return_conf_int=True)
    arima_final = processor.inverse_transform(arima_raw)
    arima_conf_int = processor.inverse_transform(arima_conf_int_raw)

    # 3. Naive Path (Uses raw training data)
    naive_final, naive_conf_int = processor.generate_seasonal_naive_forecast(train, n_periods=12)

    # 4. Compare
    arima_mae = mean_absolute_error(test, arima_final)
    naive_mae = mean_absolute_error(test, naive_final)

    print(f"SARIMA MAE: {arima_mae:.2f}")
    print(f"Naive MAE:  {naive_mae:.2f}")
    if arima_mae < naive_mae:
        print(f"Net Improvement: {naive_mae - arima_mae:.2f} ({(naive_mae - arima_mae) / naive_mae:.1%} better than Naive)")
    else:
        print(f"SARIMA did not outperform Naive. Consider revisiting model assumptions or data quality.")
        
    metrics = {
        'arima_mae': arima_mae,
        'naive_mae': naive_mae
    }

    # 5. Future Forecast (Refit on full data for actual future projection)
    full_processor = TimeSeriesProcessor(period=12)
    full_stationary = full_processor.fit_transform(ts_monthly)
    full_model = full_processor.fit_best_model(full_stationary)
    
    future_raw, future_conf_int_raw = full_model.predict(n_periods=12, return_conf_int=True)
    
    # Safely assign dates in case the index was lost during numpy transformations
    future_dates = pd.date_range(start=ts_monthly.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    future_final = pd.Series(full_processor.inverse_transform(future_raw), index=future_dates)
    future_conf_int = full_processor.inverse_transform(future_conf_int_raw)

    # 6. Visualize
    fig = plot_forecast_comparison(
        train, test, 
        arima_final, arima_conf_int,
        naive_final, naive_conf_int,
        future_forecast=future_final,
        future_ci=future_conf_int,
        title="SARIMA Forecast vs Actuals & 12-Month Future Projection"
    )
    return fig, processor, metrics

if __name__ == '__main__':
    fig, processor, metrics = run_time_series_analysis()
    fig.show()
#--------------------
#Plan
#---------------------
#"Transformation -> Differencing -> Storage.
