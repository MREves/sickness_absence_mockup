#Plan
#1. Create a class to handle preprocessing for cox proportional hazards model, including encoding, scaling, and handling of time-to-event data. This will allow us to easily prepare our synthetic dataset for survival analysis and ensure that all necessary transformations are applied consistently.
#2. create a method to fit a Cox model using the lifelines library, which will allow us to analyze the impact of various features on the time until an event (e.g., return to work after sickness absence) occurs. This method will also include functionality for evaluating model performance and interpreting results. All outputs will need to have explanations to aid interpretation 
#3. Integrate the Cox model fitting and evaluation into our existing workflow, ensuring that we can easily apply it to our synthetic dataset and visualize the results in a meaningful way. This will involve creating functions for plotting survival curves, hazard ratios, and other relevant metrics to help us understand the factors influencing sickness absence duration. Also provide the ability to enter new data and get predictions from the fitted model, including confidence intervals for those predictions at individual staff level.
#4. All outputs need to be able to be referenced within the main.py and qmd files, so we can easily include the results in our quarto reports and presentations. This will involve structuring the code in a way that allows for easy import and use of the Cox model fitting and evaluation functions across different parts of the project. Use time_series.py as a template for how to structure this code and ensure it can be easily integrated into the overall project workflow.

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import plotly.graph_objects as go
import os
import sys 
# Force the project root into the path so imports always work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.formatting import debug_print
from src.data_factory import generate_nhs_dummy_data

class CoxModel:
    def __init__(self, cols_to_drop=None, cluster_col=None):
        self.model = CoxPHFitter()
        self.fitted = False
        # Default columns to ignore during training
        self.cols_to_drop = cols_to_drop or ['staff_id', 'prev_absences', 'start_date', 'end_date', 'is_clinical']
        self.cluster_col = cluster_col
        self.training_cols = None

        # If cluster_col is set and happens to be in cols_to_drop, we must remove it 
        # so it survives the preprocessing step and can be passed to fit()
        if self.cluster_col and self.cluster_col in self.cols_to_drop:
            self.cols_to_drop.remove(self.cluster_col)
    
    def preprocess(self, df, is_training=True):
        """
        Handles preprocessing: converts Polars to Pandas, drops irrelevant columns, 
        and dynamically one-hot encodes categorical variables for the survival model.
        """
        # 1. Convert Polars to Pandas to standardize operations
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
            
        # 2. Drop explicitly defined irrelevant columns
        drop_targets = [c for c in self.cols_to_drop if c in df.columns]
        df = df.drop(columns=drop_targets)
        
        # 3. Dynamically drop any datetime columns (acts as a safety net for start/end dates)
        datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns
        df = df.drop(columns=datetime_cols)
        
        # 4. Dynamically identify and encode categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, dtype=int, drop_first=True)
            
        # 5. Align columns with training data if predicting
        if not is_training and self.training_cols is not None:
            df = df.reindex(columns=self.training_cols, fill_value=0)
        
        return df

    def fit(self, df, duration_col='duration_days', event_col='event'):
        """Fits the Cox proportional hazards model to the provided DataFrame.
        """
        processed_df = self.preprocess(df, is_training=True)
        self.training_cols = processed_df.columns.tolist()
        self.model.fit(processed_df, duration_col=duration_col, event_col=event_col, cluster_col=self.cluster_col)
        self.fitted = True
    
    def summary(self):
        """Returns the summary of the fitted Cox model."""
        if not self.fitted:
            raise ValueError("Model must be fitted before calling summary.")
        return self.model.summary
    
    def interpret_coefficients(self):
        """
        Translates hazard ratios into intuitive percentage changes in the likelihood 
        of returning to work. Highlights factors prolonging absence.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before interpreting.")
        
        summary_df = self.model.summary.copy()
        
        # HR (exp(coef)) > 1 means faster return (positive change)
        # HR < 1 means slower return (negative change, longer absence)
        summary_df['pct_change'] = (summary_df['exp(coef)'] - 1) * 100
        
        # Boolean column for statistical significance at 0.05
        summary_df['is_significant'] = summary_df['p'] < 0.05
        
        # Sort by HR ascending (factors most strongly preventing return at the top)
        summary_df = summary_df.sort_values(by='exp(coef)', ascending=True)
        
        print("\n--- Wellbeing Strategy Insight: Impact on Return to Work ---")
        interpretations = []
        for feature, row in summary_df.iterrows():
            pct = row['pct_change']
            sig_flag = "*" if row['is_significant'] else ""
            
            if pct < 0:
                direction = "Decreases"
                impact = "PROLONGS absence"
            else:
                direction = "Increases"
                impact = "SHORTENS absence"
                
            explanation = f"{direction} the likelihood of returning to work by {abs(pct):.1f}% ({impact})"
            interpretations.append(explanation)
            
            print(f"Feature '{feature}': {explanation} {sig_flag}")
                  
        summary_df['interpretation'] = interpretations
        print("* indicates statistical significance (p < 0.05)\n")
        
        # Return the simplified dataframe for potential Quarto tables
        return summary_df[['exp(coef)', 'pct_change', 'p', 'is_significant', 'interpretation']]
    
    def plot_survival_curves(self, df, covariates, title="Partial Effects on Survival"):
        """Plots survival curves varying specific covariates while holding others at baseline.
        Returns a dictionary mapping covariates to their respective Plotly figure objects."""
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting survival curves.")
        
        # Ensure we're working with Pandas to check original columns safely
        if hasattr(df, "to_pandas"):
            df_pd = df.to_pandas()
        else:
            df_pd = df
            
        processed_df = self.preprocess(df_pd, is_training=False)
        plots = {}
        
        # Create a baseline profile using the mean of the features
        baseline = processed_df.mean().to_frame().T
        
        for covariate in covariates:
            synthetic_df = None
            
            # 1. Check if it's an original categorical column (e.g. 'role' instead of 'role_Manager')
            if covariate in df_pd.columns and df_pd[covariate].dtype in ['object', 'category', 'string']:
                values_to_plot = sorted(df_pd[covariate].dropna().unique())
                synthetic_df = pd.concat([baseline] * len(values_to_plot), ignore_index=True)
                
                for i, val in enumerate(values_to_plot):
                    # Reset all dummy columns for this covariate to 0
                    dummy_cols = [c for c in processed_df.columns if c.startswith(f"{covariate}_")]
                    for col in dummy_cols:
                        synthetic_df.loc[i, col] = 0
                        
                    # Set the specific dummy column to 1 (if it wasn't dropped as the baseline)
                    target_col = f"{covariate}_{val}"
                    if target_col in processed_df.columns:
                        synthetic_df.loc[i, target_col] = 1
                        
            # 2. Check if it's a continuous or already processed binary column
            elif covariate in processed_df.columns:
                unique_vals = sorted(processed_df[covariate].dropna().unique())
                if len(unique_vals) <= 5:
                    values_to_plot = unique_vals
                else:
                    values_to_plot = [processed_df[covariate].quantile(q) for q in [0.10, 0.25, 0.50, 0.75, 0.90]]
                
                synthetic_df = pd.concat([baseline] * len(values_to_plot), ignore_index=True)
                synthetic_df[covariate] = values_to_plot
                
            if synthetic_df is not None:
                # Predict survival curves for the synthetic data
                survival_curves = self.model.predict_survival_function(synthetic_df)
                
                fig = go.Figure()
                
                # Default Plotly color sequence
                colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
                
                # Calculate standard errors for the predicted curves (partial confidence intervals)
                # This isolates the uncertainty of the covariate effects relative to the baseline
                features = self.model.params_.index.tolist()
                X_centered = (synthetic_df[features] - processed_df[features].mean()).astype(float)
                se = np.sqrt((X_centered.values @ self.model.variance_matrix_.values * X_centered.values).sum(axis=1))
                
                for i, val in enumerate(values_to_plot):
                    label_val = round(val, 2) if isinstance(val, (int, float, np.number)) else val
                    color = colors[i % len(colors)]
                    
                    s_curve = survival_curves.iloc[:, i]
                    
                    # 95% Confidence Intervals for survival: S(t)^exp(+/- 1.96 * SE)
                    s_lower = s_curve ** np.exp(1.96 * se[i])
                    s_upper = s_curve ** np.exp(-1.96 * se[i])
                    
                    # CI Fill (Upper bound - invisible)
                    fig.add_trace(go.Scatter(
                        x=s_curve.index, 
                        y=s_upper, 
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        legendgroup=f"group_{i}",
                        hoverinfo='skip'
                    ))
                    
                    # CI Fill (Lower bound - fills area up to the upper bound)
                    rgba_color = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)"
                    fig.add_trace(go.Scatter(
                        x=s_curve.index, 
                        y=s_lower, 
                        mode='lines',
                        fill='tonexty',
                        fillcolor=rgba_color,
                        line=dict(width=0),
                        showlegend=False,
                        legendgroup=f"group_{i}",
                        hoverinfo='skip'
                    ))

                    # Main curve
                    fig.add_trace(go.Scatter(
                        x=s_curve.index, 
                        y=s_curve, 
                        mode='lines',
                        name=f"{covariate} = {label_val}",
                        line=dict(color=color, width=2),
                        legendgroup=f"group_{i}"
                    ))
                
                fig.update_layout(
                    title=f"Survival Curve by {covariate.replace('_', ' ').title()}",
                    xaxis_title="Time (Days)",
                    yaxis_title="Survival Probability (Still Absent)",
                    template="plotly_white",
                    hovermode="x unified",
                    margin=dict(r=80), # Increase right margin buffer
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.98,
                        xanchor="right",
                        x=0.90,        # Move it further left (from 0.95 to 0.90)
                        bgcolor="rgba(255, 255, 255, 0.8)"
                    )
                )
                
                plots[covariate] = fig
                
        return plots
    
    def predict(self, df):
        """Predicts survival probabilities for new data based on the fitted model."""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions.")
        processed_df = self.preprocess(df, is_training=False)
        return self.model.predict_survival_function(processed_df)
    
    def predict_expected_duration(self, df):
        """
        Predicts the expected time to return to work for individuals.
        Returns the median expected days, along with an 80% prediction interval 
        (the window between the 10th and 90th percentiles of their return probability).
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions.")
        
        processed_df = self.preprocess(df, is_training=False)
        
        # Median return time (50% chance of returning by this day)
        median_days = self.model.predict_median(processed_df)
        
        # 80% Prediction Interval:
        # Time when survival drops to 0.9 (10% chance they returned - early boundary)
        early_return = self.model.predict_percentile(processed_df, p=0.9)
        # Time when survival drops to 0.1 (90% chance they returned - late boundary)
        late_return = self.model.predict_percentile(processed_df, p=0.1)
        
        df_out = pd.DataFrame({
            'predicted_median_days': median_days,
            'early_return_boundary_days (10%)': early_return,
            'late_return_boundary_days (90%)': late_return
        })
        
        interpretations = []
        for _, row in df_out.iterrows():
            median = row['predicted_median_days']
            early = row['early_return_boundary_days (10%)']
            late = row['late_return_boundary_days (90%)']
            
            median_val = f"{median:.0f}" if pd.notna(median) and median != np.inf else "unknown"
            early_val = f"{early:.0f}" if pd.notna(early) and early != np.inf else "unknown"
            
            if late == np.inf or pd.isna(late):
                interp = f"Expected to be absent for {median_val} days. There is a >10% chance this absence may be indefinite based on historical patterns."
            else:
                interp = f"Expected to be absent for {median_val} days. There is an 80% probability their actual return will fall between day {early_val} and day {late:.0f}."
            interpretations.append(interp)
            
        df_out['interpretation'] = interpretations
        return df_out
    
    def evaluate(self, df, duration_col='duration_days', event_col='event'):
        """
        Evaluates the model using both Concordance Index (survival standard) 
        and Mean Absolute Error (regression standard, applied to uncensored data only).
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before evaluation.")
            
        processed_df = self.preprocess(df, is_training=False)
        
        # 1. Concordance Index (C-Index)
        # This is the standard survival metric. It measures if the model correctly ranks 
        # who will return to work first. 0.5 is random guessing, 1.0 is perfect prediction.
        c_index = self.model.score(processed_df, scoring_method="concordance_index")
        
        # 2. Mean Absolute Error (MAE) on Uncensored Data
        # We can only calculate MAE for people who ACTUALLY returned. 
        uncensored_df = processed_df[processed_df[event_col] == 1]
        
        if len(uncensored_df) > 0:
            predicted_medians = self.model.predict_median(uncensored_df)
            
            # Isolate finite predictions (ignoring 'inf' where the model thinks they may never return)
            is_finite = np.isfinite(predicted_medians).values.flatten()
            
            actual_durations = uncensored_df[duration_col].values[is_finite]
            valid_preds = predicted_medians.values.flatten()[is_finite]
            
            mae = np.abs(actual_durations - valid_preds).mean()
        else:
            mae = None
            
        return {
            'concordance_index': c_index,
            'mae_uncensored_days': mae
        }

    #test class with synthetic data
if __name__ == "__main__":
    ts_df, staff_df = generate_nhs_dummy_data()
    cox_model = CoxModel(cluster_col='staff_id')
    cox_model.fit(staff_df)
    
    # Interprets the model mathematically and conversationally
    insights_df = cox_model.interpret_coefficients()
    
    # Dynamically plot using original logical features before one-hot encoding
    base_features = [c for c in staff_df.columns if c not in cox_model.cols_to_drop and c not in ['duration_days', 'event']]
    plots = cox_model.plot_survival_curves(staff_df, covariates=base_features)
    
    for cov, fig in plots.items():
        fig.show() # Display the interactive Plotly figures
    
    print("\n--- Individual Absence Duration Predictions (First 5 Staff) ---")
    duration_predictions = cox_model.predict_expected_duration(staff_df.head(5))
    print(duration_predictions)
    
    print("\n--- Model Evaluation ---")
    metrics = cox_model.evaluate(staff_df)
    print(f"Concordance Index: {metrics['concordance_index']:.3f} (0.5 is random, 1.0 is perfect)")
    print(f"Mean Absolute Error (Uncensored only): {metrics['mae_uncensored_days']:.1f} days")
    
    print("")
    print("--- Model Coefficients and Insights ---")
    print(insights_df)
