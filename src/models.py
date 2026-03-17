
import polars as pl
import pandas as pd
import plotly.graph_objects as go

# Import our custom utility classes
from src.utils.cph import CoxModel
from src.time_series import run_time_series_analysis

def get_dashboard_survival_outputs(staff_df: pl.DataFrame):
    """
    Orchestrates the Cox Proportional Hazards pipeline for the dashboard.
    Returns the fitted model, insights dataframe, and a dictionary of plots.
    """
    cox = CoxModel()
    cox.fit(staff_df)
    
    insights_df = cox.interpret_coefficients()
    
    # Identify significant dummies
    sig_dummies = insights_df[insights_df['is_significant']].index.tolist()
    
    # Identify original base features before one-hot encoding
    all_base_features = [c for c in staff_df.columns if c not in cox.cols_to_drop and c not in ['duration_days', 'event']]
    significant_base_features = []
    
    for feature in all_base_features:
        # A feature is kept if it matches directly, OR if a dummy subclass derived from it was significant
        if feature in sig_dummies or any(d.startswith(f"{feature}_") for d in sig_dummies):
            significant_base_features.append(feature)
            
    plots = cox.plot_survival_curves(staff_df, covariates=significant_base_features)
    
    metrics = cox.evaluate(staff_df)
    
    return cox, insights_df, plots, metrics

def get_dashboard_forecast_outputs():
    """
    Orchestrates the Time Series Forecasting pipeline for the dashboard.
    Returns the generated comparison figure and evaluation metrics.
    """
    fig, processor, metrics = run_time_series_analysis()
    return fig, metrics