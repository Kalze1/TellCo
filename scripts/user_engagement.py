import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#  Aggregate user engagement metrics
def aggregate_engagement_metrics(df):
    # Ensure 'Total DL (Bytes)' and 'Total UL (Bytes)' are numeric
    df['Total DL (Bytes)'] = pd.to_numeric(df['Total DL (Bytes)'], errors='coerce')
    df['Total UL (Bytes)'] = pd.to_numeric(df['Total UL (Bytes)'], errors='coerce')
    
    # Use the correct column name if 'MSISDN' is different, e.g., 'MSISDN/Number'
    if 'MSISDN/Number' in df.columns:
        user_metrics = df.groupby('MSISDN/Number').agg(
            session_count=('Bearer Id', 'count'),  # Frequency of sessions
            total_duration=('Dur. (ms)', 'sum'),   # Total session duration
            total_traffic_dl=('Total DL (Bytes)', 'sum'),  # Total download traffic
            total_traffic_ul=('Total UL (Bytes)', 'sum')   # Total upload traffic
        ).reset_index()
        
        # Sum download and upload traffic to get total traffic per user
        user_metrics['total_traffic'] = user_metrics['total_traffic_dl'] + user_metrics['total_traffic_ul']
    else:
        raise KeyError("The column 'MSISDN' or 'MSISDN/Number' is missing from the dataset.")
    
    return user_metrics



# Normalize engagement metrics and apply k-means clustering
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

#  Compute statistics for each cluster
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

# Aggregate traffic per application
def aggregate_traffic_per_app(df):
    """
    Aggregates total traffic per application across all users.
    """
    apps = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)',
            'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    
    # Sum traffic per application for all users
    app_traffic = df[apps].sum().reset_index()
    app_traffic.columns = ['Application', 'Total Traffic (Bytes)']
    
    return app_traffic


# Plot top 3 most used applications
def plot_top_3_apps(app_traffic):
    """
    Plots the top 3 most used applications by traffic.
    """
    # Ensure 'app_traffic' is a DataFrame
    if not isinstance(app_traffic, pd.DataFrame):
        raise TypeError("Expected a DataFrame for 'app_traffic', but got something else.")
    
    # Sort by 'Total Traffic (Bytes)' and get the top 3 applications
    top_3_apps = app_traffic.sort_values(by='Total Traffic (Bytes)', ascending=False).head(3)
    
    # Plot the top 3 apps
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Application', y='Total Traffic (Bytes)', data=top_3_apps)
    plt.title('Top 3 Most Used Applications by Traffic')
    plt.xlabel('Application')
    plt.ylabel('Total Traffic (Bytes)')
    plt.show()



#  Use the elbow method to find the optimal k
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

#  Evaluate cluster quality with silhouette score
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
