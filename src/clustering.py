# Purpose:
#   - Clustering of the data
#   - Identify groups of staff with similar characteristics and absence patterns
#.  - This can help in understanding different risk profiles and tailoring interventions
#.  - This may compliment survival analysis outputs, e.g. by showing which groups have similar absence durations or return-to-work patterns

#   - Methods:
#       - K-Means Clustering: A popular clustering algorithm that partitions data into K distinct clusters based on feature similarity.
#       - Hierarchical Clustering: Builds a hierarchy of clusters, which can be visualized as a dendrogram. It does not require specifying the number of clusters in advance.
#       - DBSCAN: A density-based clustering algorithm that identifies clusters based on the density of data points, which can be useful for identifying outliers and clusters of varying shapes and sizes.


# Plan:
#   1. Data Preparation:
#       - Import dummy data generated from the data factory
#       - Select relevant features for clustering (e.g., age, gender, role, tenure, previous absences, duration of absence)
#       - consider implication of staff who have multiple episodes of absence and how to represent this in the data (e.g., aggregate features, separate rows for each episode)
#       - Standardize the features to ensure they are on the same scale
#   2. Clustering Analysis:
#       - Apply K-Means clustering to identify distinct groups of staff based on their characteristics and absence patterns
#       - Use the silhouette score to evaluate the optimal number of clusters
#       - Apply DBSCAN to identify any outliers or clusters of varying shapes and sizes
#       - Evaluate K-Means and DBSCAN to see which provides more meaningful clusters in the context of the data using silhouette scores and visualizations
#   3. Visualization:
#       - Create visualizations to show the clusters and their characteristics (e.g., scatter plots, cluster centers, dendrograms for hierarchical clustering)
#       - use PCA or t-SNE for dimensionality reduction to visualize clusters in 2D or 3D space
#   4. Interpretation:
#       - Analyze the characteristics of each cluster to understand the different risk profiles and absence patterns


# ----------------------------
# Import necessary libraries and modules
# ----------------------------

import polars as pl
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Force the project root into the path so imports always work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_factory import generate_nhs_dummy_data

# ----------------------------
# Create Class for clustering analysis
# ----------------------------
class StaffClustering:
    """
    Class to handle preprocessing, clustering, visualization, and interpretation
    of staff sickness absence data to identify different risk profiles.
    """
    def __init__(self, random_state=42):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=random_state)
        self.kmeans = None
        self.dbscan = None
        self.kmeans_score = -1.0
        self.dbscan_score = -1.0
        self.best_model_type = None
        self.best_k = None
        self.best_eps = None
        self.best_min_samples = None
        self.random_state = random_state
        self.processed_df = None
        self.original_df = None
        self.scaled_data = None
        self.pca_data = None
        self.model_scores_df = None

    def preprocess(self, df):
        """
        Prepares the dataset for clustering by selecting features, handling
        categorical variables via one-hot encoding, and scaling the data.
        """
        if hasattr(df, "to_pandas"):
            self.original_df = df.to_pandas()
        else:
            self.original_df = df.copy()

        # Features relevant to staff absence risk profiles
        features = ['age', 'gender', 'role', 'imd_quintile', 'tenure_years', 'prev_absences', 'duration_days']
        df_selected = self.original_df[features].copy()

        # Encode categoricals dynamically
        cat_cols = df_selected.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        self.processed_df = pd.get_dummies(df_selected, columns=cat_cols, drop_first=True)

        # Standardize the features so they are on the same scale
        self.scaled_data = self.scaler.fit_transform(self.processed_df)
        return self.scaled_data

    def find_optimal_kmeans(self, k_range=range(2, 20)):
        """
        Tests a range of K values for K-Means and selects the optimal number
        of clusters based on the highest silhouette score.
        """
        best_score = -1
        best_model = None

        print("Evaluating K-Means for optimal K...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
            labels = kmeans.fit_predict(self.scaled_data)
            score = silhouette_score(self.scaled_data, labels)

            if score > best_score:
                best_score = score
                best_model = kmeans
                self.best_k = k

        self.kmeans = best_model
        self.kmeans_score = best_score
        print(f"Optimal K found: {self.best_k} (Silhouette Score: {best_score:.4f})")
        return self.best_k, best_score

    def find_optimal_dbscan(self, eps_range=np.arange(0.5, 4.0, 0.5), min_samples_range=range(3, 10)):
        """
        Tests a grid of eps and min_samples values for DBSCAN and selects the
        optimal combination based on the highest silhouette score.
        """
        best_score = -1.0
        best_model = None
        best_labels = None
        
        print("Evaluating DBSCAN for optimal hyperparameters...")
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan.fit(self.scaled_data)
                labels = dbscan.labels_
                
                # Silhouette score is only defined if number of clusters > 1
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters > 1:
                    score = silhouette_score(self.scaled_data, labels)
                    if score > best_score:
                        best_score = score
                        best_model = dbscan
                        best_labels = labels
                        self.best_eps = eps
                        self.best_min_samples = min_samples

        if best_model is not None:
            self.dbscan = best_model
            self.dbscan_score = best_score
            n_outliers = list(best_labels).count(-1)
            n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            print(f"Optimal DBSCAN found: eps={self.best_eps:.1f}, min_samples={self.best_min_samples} ")
            print(f"(Silhouette Score: {self.dbscan_score:.4f}, Clusters: {n_clusters}, Outliers: {n_outliers})")
            return best_labels
        else:
            print("No valid DBSCAN clusters found across the provided ranges.")
            self.dbscan_score = -1.0
            # Fallback
            self.dbscan = DBSCAN(eps=2.0, min_samples=5).fit(self.scaled_data)
            return self.dbscan.labels_

    def compare_models(self):
        """
        Compares the silhouette scores of K-Means and DBSCAN to select the preferred model.
        """
        self.model_scores_df = pd.DataFrame({
            'Model': ['K-Means', 'DBSCAN'],
            'Silhouette Score': [self.kmeans_score, self.dbscan_score]
        })
        
        print("\n--- Model Evaluation & Comparison ---")
        # Print the dataframe directly to the terminal
        print(self.model_scores_df.to_string(index=False))
        
        if self.kmeans_score >= self.dbscan_score:
            self.best_model_type = 'kmeans'
            print("\n=> Preferred Model: K-Means")
        else:
            self.best_model_type = 'dbscan'
            print("\n=> Preferred Model: DBSCAN")
            
        return self.best_model_type

    def reduce_dimensions(self):
        """Reduces dimensions to 2D using PCA for visualization."""
        self.pca_data = self.pca.fit_transform(self.scaled_data)
        return self.pca_data

    def plot_clusters(self, model_type=None):
        """
        Plots the identified clusters on a 2D PCA scatter plot.
        """
        if model_type is None:
            model_type = self.best_model_type or 'kmeans'
            
        if self.pca_data is None:
            self.reduce_dimensions()

        plot_df = pd.DataFrame(self.pca_data, columns=['PCA1', 'PCA2'])

        if model_type == 'kmeans' and self.kmeans is not None:
            plot_df['Cluster'] = [f"Cluster {c}" for c in self.kmeans.labels_]
            title = f"Staff Risk Profiles (K-Means, K={self.best_k})"
        elif model_type == 'dbscan' and self.dbscan is not None:
            plot_df['Cluster'] = [f"Cluster {c}" if c != -1 else "Outlier" for c in self.dbscan.labels_]
            title = "Staff Risk Profiles (DBSCAN)"
        else:
            raise ValueError(f"Model {model_type} not fitted.")

        # Append some original features for hover data
        plot_df['Age'] = self.original_df['age'].values
        plot_df['Duration'] = self.original_df['duration_days'].values

        fig = px.scatter(
            plot_df, x='PCA1', y='PCA2', color='Cluster',
            hover_data=['Age', 'Duration'],
            title=title, template='plotly_white',
            opacity=0.7, color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig.update_layout(margin=dict(r=80, t=50, b=50, l=50))
        return fig

    def interpret_clusters(self, model_type=None):
        """
        Groups the original unscaled data by cluster to extract the
        characteristics and meaning behind each cluster.
        """
        if model_type is None:
            model_type = self.best_model_type or 'kmeans'
            
        if model_type == 'kmeans' and self.kmeans is not None:
            labels = self.kmeans.labels_
        elif model_type == 'dbscan' and self.dbscan is not None:
            labels = self.dbscan.labels_
        else:
            raise ValueError(f"Model {model_type} not fitted.")

        analysis_df = self.original_df.copy()
        analysis_df['Cluster'] = labels

        num_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'staff_id' in num_cols: num_cols.remove('staff_id')
        if 'Cluster' in num_cols: num_cols.remove('Cluster')
        if 'event' in num_cols: num_cols.remove('event')

        cat_cols = analysis_df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

        # Calculate means for numeric data and format
        numeric_summary = analysis_df.groupby('Cluster')[num_cols].mean().round(1)

        # Calculate the most frequent value (mode) for categorical features
        if cat_cols:
            cat_summary = analysis_df.groupby('Cluster')[cat_cols].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            summary = pd.concat([numeric_summary, cat_summary], axis=1)
        else:
            summary = numeric_summary

        # Append Cluster Size
        summary['Staff_Count'] = analysis_df.groupby('Cluster').size()

        return summary

# ----------------------------
# Create functions for clustering analysis
# ----------------------------
def run_clustering_analysis(df=None):
    """
    Orchestrates the clustering analysis workflow: preprocessing,
    optimal K finding, clustering, and interpreting.
    """
    if df is None:
        _, df = generate_nhs_dummy_data()

    clustering = StaffClustering()
    clustering.preprocess(df)

    # 1. Fit K-Means
    clustering.find_optimal_kmeans()

    # 2. Fit DBSCAN with Grid Search
    clustering.find_optimal_dbscan()

    # 3. Compare Models
    best_model = clustering.compare_models()

    # 4. Generate Visualization
    fig = clustering.plot_clusters(model_type=best_model)

    # 5. Generate Interpretation
    insights = clustering.interpret_clusters(model_type=best_model)

    return clustering, fig, insights

if __name__ == "__main__":
    # Test script execution
    model, fig, insights = run_clustering_analysis()
    
    print(f"\n--- Cluster Interpretation ({model.best_model_type.upper()}) ---")
    print(insights)
    
    fig.show()
