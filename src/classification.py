# Purpose:
#   - Identifying features associated with an absence being long-term (>= some set value of days)

# Plan
# 1. Define a threshold for long-term absence (e.g., 30 days)
# 2. Create a binary target variable indicating whether an absence is long-term or not
# 3. Select relevant features for classification (e.g., age, department, reason for absence, etc.)
# 4. scale the features if necessary
# 5. Split the data into training and testing sets and hold out a validation set
# 6. Train a classification model (e.g., logistic regression, random forest, XGBoost etc.)
# 7. Address class imbalance if necessary (e.g., using SMOTE, class weights, etc.)
# 8. Evaluate the model's performance using appropriate metrics (e.g., accuracy, precision, recall, F1-score)
# 9. Optimise the model using techniques like cross-validation and hyperparameter tuning
# 10. Interpret the model to identify key features associated with long-term absences (e.g., feature importance, SHAP values, etc.)

#--------------------
# Import necessary libraries
#--------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
#from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

#--------------------
# Load the dummy dataset
#--------------------
# For demonstration purposes, we'll create a dummy dataset
# Force the project root into the path so imports always work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_factory import generate_nhs_dummy_data
ts_df, staff_df = generate_nhs_dummy_data()
cols = ['staff_id', 'age', 'gender', 'role', 'imd_quintile', 'tenure_years', 'prev_absences', 'is_clinical', 'duration_days', 'event']
staff_df = staff_df.drop(['staff_id', 'event'])

#--------------------
# define functions for classification
#--------------------

def create_target_variable(df, absenceDurationColumn, threshold=30):
    """
    Creates a binary target variable indicating whether an absence is long-term or not.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the absence data.
    absenceDurationColumn (str): The name of the column in df that contains the duration of absences.
    threshold (int): The number of days that defines a long-term absence (default is 30).
    
    Returns:
    pd.DataFrame: A DataFrame with an additional binary target variable column named 'long_term_absence'.
    """
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    df['long_term_absence'] = (df[absenceDurationColumn] >= threshold)
    return df


#--------------------
# Create Classes
#--------------------

class AbsenceClassifierPipeline:
    def __init__(self, df, target_column, feature_columns, absence_threshold=30):
        self.df = df
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.absence_threshold = absence_threshold
        self.cat_cols = []
        self.num_cols = []
        self.preprocessor = None
        
        # Datasets
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Models and evaluations
        self.base_models = {}
        self.tuned_models = {}
        self.best_thresholds = {}
        self.evaluation_results = None
    
    def preprocess_and_split(self, test_size=0.2, val_size=0.25, random_state=42):
        """
        Splits the data into train, validation, and test sets, and defines the 
        ColumnTransformer to handle scaling and one-hot encoding without data leakage.
        """
        if hasattr(self.df, "to_pandas"):
            df = self.df.to_pandas()
        else:
            df = self.df.copy()
            
        # Handle missing values
        df.dropna(subset=self.feature_columns + [self.target_column], inplace=True)
        
        # Identify categorical and numerical columns dynamically
        self.cat_cols = df[self.feature_columns].select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        self.num_cols = [col for col in self.feature_columns if col not in self.cat_cols]
        
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # Split into train+val and test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        # Split train+val into train and val (if test_size=0.2 and val_size=0.25, we get a 60/20/20 split)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp)
        
        # Build the preprocessor (scaling and encoding applied later dynamically in the pipeline)
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_cols)
            ])

    def add_model(self, model_name, model_instance):
        """Registers a base model for tuning and evaluation."""
        if self.preprocessor is None:
            raise ValueError("Data must be preprocessed and split before adding models.")
            
        # Wrap the preprocessor and the classifier inside a Pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', model_instance)
        ])
        
        self.base_models[model_name] = pipeline

    def tune_model(self, model_name, param_grid, cv=5, scoring='f1'):
        """Performs GridSearchCV to find the best hyperparameters for a given model."""
        if model_name not in self.base_models:
            raise ValueError(f"Model '{model_name}' has not been added.")
            
        print(f"Tuning {model_name}...")
        grid_search = GridSearchCV(
            estimator=self.base_models[model_name], 
            param_grid=param_grid, 
            cv=cv, 
            scoring=scoring, 
            n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        self.tuned_models[model_name] = grid_search.best_estimator_
        print(f"[{model_name}] Best params: {grid_search.best_params_}")
        return self.tuned_models[model_name]

    def optimize_threshold(self, model_name, metric='f1'):
        """Finds the optimal decision threshold for a tuned model using the validation set."""
        if model_name not in self.tuned_models:
            raise ValueError(f"Model '{model_name}' must be tuned before threshold optimization.")
            
        model = self.tuned_models[model_name]
        
        if not hasattr(model, "predict_proba"):
            print(f"Model '{model_name}' does not support predict_proba. Using default threshold 0.5.")
            self.best_thresholds[model_name] = 0.5
            return 0.5
            
        y_val_probs = model.predict_proba(self.X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(self.y_val, y_val_probs)
        
        best_threshold = 0.5
        best_score = -1
        
        # Calculate metric across all thresholds
        for threshold in thresholds:
            y_val_pred = (y_val_probs >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(self.y_val, y_val_pred, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(self.y_val, y_val_pred)
            elif metric == 'precision':
                score = precision_score(self.y_val, y_val_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(self.y_val, y_val_pred, zero_division=0)
            else:
                raise ValueError(f"Metric '{metric}' is not supported for threshold optimization.")
                
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
        self.best_thresholds[model_name] = best_threshold
        print(f"[{model_name}] Optimal Threshold ({metric}): {best_threshold:.4f} (Score: {best_score:.4f})")
        return best_threshold

    def evaluate_all_models(self):
        """Evaluates all tuned models on the held-out test set using optimized thresholds."""
        results = []
        
        for name, model in self.tuned_models.items():
            threshold = self.best_thresholds.get(name, 0.5)
            
            if hasattr(model, "predict_proba"):
                y_test_probs = model.predict_proba(self.X_test)[:, 1]
                y_test_pred = (y_test_probs >= threshold).astype(int)
                roc_auc = roc_auc_score(self.y_test, y_test_probs)
            else:
                y_test_pred = model.predict(self.X_test)
                roc_auc = np.nan
                
            acc = accuracy_score(self.y_test, y_test_pred)
            prec = precision_score(self.y_test, y_test_pred, zero_division=0)
            rec = recall_score(self.y_test, y_test_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_test_pred, zero_division=0)
            
            results.append({
                'Model': name,
                'Threshold': round(threshold, 4),
                'Accuracy': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1-Score': round(f1, 4),
                'ROC-AUC': round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A"
            })
            
        self.evaluation_results = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
        return self.evaluation_results

    def get_feature_importances(self, model_name):
        """Returns the feature importances or coefficients for the specified model."""
        if model_name not in self.tuned_models:
            raise ValueError(f"Model '{model_name}' has not been tuned.")
            
        pipeline = self.tuned_models[model_name]
        classifier = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Reconstruct feature names from the column transformer
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                feature_names.extend(transformer.get_feature_names_out(cols))
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importances = np.abs(classifier.coef_[0])
        else:
            print(f"Model '{model_name}' does not support feature importances extraction.")
            return None
            
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        return importance_df

#--------------------
# Example usage
#--------------------

if __name__ == "__main__":
    absence_threshold = 30
    
    # 1. Create target variable
    staff_df = create_target_variable(staff_df, absenceDurationColumn='duration_days', threshold=absence_threshold)
    
    # 2. Initialize pipeline
    feature_cols = ['age', 'gender', 'role', 'imd_quintile', 'tenure_years', 'prev_absences', 'is_clinical']  # Example feature columns
    pipeline = AbsenceClassifierPipeline(df=staff_df, target_column='long_term_absence', feature_columns=feature_cols, absence_threshold=absence_threshold)
    
    # 3. Preprocess and split data
    pipeline.preprocess_and_split()
    
    # 4. Add models to the pipeline
    pipeline.add_model('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced'))
    pipeline.add_model('Random Forest', RandomForestClassifier(random_state=42, class_weight='balanced'))
    pipeline.add_model('Gradient Boosting', GradientBoostingClassifier(random_state=42))
    
    # 5. Tune models with GridSearchCV
    lr_params = {'classifier__C': [0.01, 0.1, 1, 10]}
    rf_params = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 10, 20]}
    gb_params = {'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [0.01, 0.1]}
    
    pipeline.tune_model('Logistic Regression', lr_params)
    pipeline.tune_model('Random Forest', rf_params)
    pipeline.tune_model('Gradient Boosting', gb_params)
    
    # 6. Optimize thresholds for each model
    for model_name in pipeline.tuned_models.keys():
        pipeline.optimize_threshold(model_name, metric='f1')
    
    # 7. Evaluate all models on the test set
    evaluation_results = pipeline.evaluate_all_models()
    print(evaluation_results)
    
    # 8. Get feature importances for the best model (based on F1-Score)
    best_model_name = evaluation_results.iloc[0]['Model']
    feature_importances = pipeline.get_feature_importances(best_model_name)
    #print(f"Feature Importances for {best_model_name}:\n", feature_importances)