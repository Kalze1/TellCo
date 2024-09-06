import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Aggregate user engagement metrics
def aggregate_engagement_metrics(df):
    """
    Aggregates the following metrics per user (MSISDN):
    - Number of sessions
    - Total session duration
    - Total session traffic (DL + UL)
    """
    user_metrics = df.groupby('MSISDN').agg(
        session_count=('Bearer Id', 'count'),  # Frequency of sessions
        total_duration=('Dur. (ms)', 'sum'),   # Total session duration
        total_traffic=('Total DL (Bytes)', 'sum') + df['Total UL (Bytes)'],  # Total traffic (DL + UL)
    ).reset_index()
    
    return user_metrics

# 2. Normalize engagement metrics and apply k-means clustering
def normalize_and_cluster(df, k=3):
    """
    Normalizes the engagement metrics and applies k-means clustering.
    """
    # Normalize the engagement metrics using Min-Max scaling
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[['session_count', 'total_duration', 'total_traffic']])
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(normalized_data)
    
    return df, kmeans

# 3. Compute statistics for each cluster
def compute_cluster_stats(df):
    """
    Computes the min, max, average, and total metrics for each cluster.
    """
    cluster_stats = df.groupby('cluster').agg(
        min_sessions=('session_count', 'min'),
        max_sessions=('session_count', 'max'),
        avg_sessions=('session_count', 'mean'),
        total_sessions=('session_count', 'sum'),
        
        min_duration=('total_duration', 'min'),
        max_duration=('total_duration', 'max'),
        avg_duration=('total_duration', 'mean'),
        total_duration=('total_duration', 'sum'),
        
        min_traffic=('total_traffic', 'min'),
        max_traffic=('total_traffic', 'max'),
        avg_traffic=('total_traffic', 'mean'),
        total_traffic=('total_traffic', 'sum')
    ).reset_index()
    
    return cluster_stats

# 4. Aggregate traffic per application
def aggregate_traffic_per_app(df):
    """
    Aggregates user traffic per application and reports the top 10 most engaged users.
    """
    apps = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)',
            'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    
    # Sum traffic per application for each user
    app_traffic = df.groupby('MSISDN').agg({app: 'sum' for app in apps}).reset_index()
    
    # Find top 10 most engaged users per application
    top_10_users_per_app = {app: app_traffic.nlargest(10, app)[['MSISDN', app]] for app in apps}
    
    return top_10_users_per_app

# 5. Plot top 3 most used applications
def plot_top_3_apps(app_traffic):
    """
    Plots the top 3 most used applications by traffic.
    """
    # Sum traffic for all users per application
    total_traffic_per_app = app_traffic[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)',
                                         'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']].sum()
    
    # Sort and get the top 3 apps
    top_3_apps = total_traffic_per_app.sort_values(ascending=False).head(3)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_3_apps.index, y=top_3_apps.values)
    plt.title('Top 3 Most Used Applications by Traffic')
    plt.xlabel('Application')
    plt.ylabel('Total Traffic (Bytes)')
    plt.show()

# 6. Use the elbow method to find the optimal k
def elbow_method(df):
    """
    Uses the elbow method to find the optimal number of clusters (k).
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[['session_count', 'total_duration', 'total_traffic']])
    
    inertia = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_data)
        inertia.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia, 'bo-')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Return optimal k based on visual inspection of the elbow plot
    return inertia

# 7. Evaluate cluster quality with silhouette score
def silhouette_analysis(df, k):
    """
    Compute and return the silhouette score for k clusters.
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[['session_count', 'total_duration', 'total_traffic']])
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_data)
    
    score = silhouette_score(normalized_data, cluster_labels)
    
    return score
