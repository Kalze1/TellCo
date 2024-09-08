import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns




def aggregate_per_customer(df):
    """Aggregate data per customer to get average metrics."""
    grouped_df = df.groupby('MSISDN/Number').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'Handset Type': 'first'  
    }).reset_index()

    # Rename columns for clarity
    grouped_df.columns = [
        'Customer', 'Avg TCP DL Retrans (Bytes)', 'Avg TCP UL Retrans (Bytes)',
        'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Throughput DL (kbps)',
        'Avg Throughput UL (kbps)', 'Handset Type'
    ]

    return grouped_df





def compute_top_bottom_frequent(df, column, top_n=10):
    """Compute the top, bottom, and most frequent values for a given column."""
    
    top_values = df[column].nlargest(top_n)
    bottom_values = df[column].nsmallest(top_n)
    frequent_values = df[column].value_counts().head(top_n)
    
    return top_values, bottom_values, frequent_values

def compute_top_bottom_frequent_main(df):
    """Main function to compute and list top, bottom, and frequent values for TCP, RTT, and Throughput."""
    
    # Columns of interest
    columns_of_interest = [
        'TCP DL Retrans. Vol (Bytes)',
        'TCP UL Retrans. Vol (Bytes)',
        'Avg RTT DL (ms)',
        'Avg RTT UL (ms)',
        'Avg Bearer TP DL (kbps)',
        'Avg Bearer TP UL (kbps)'
    ]
    
    results = {}
    
    # Iterate through each column and compute the required statistics
    for col in columns_of_interest:
        top_values, bottom_values, frequent_values = compute_top_bottom_frequent(df, col)
        results[col] = {
            'Top Values': top_values,
            'Bottom Values': bottom_values,
            'Most Frequent Values': frequent_values
        }
    
    return results



def analyze_throughput_tcp_retransmission(df):
    """Compute and report the distribution of average throughput and TCP retransmission per handset type."""
    
    # Group by 'Handset Type' and calculate the average throughput and TCP retransmission
    handset_stats = df.groupby('Handset Type').agg({
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean'
    }).reset_index()

    # Plot the distribution of average throughput per handset type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Avg Bearer TP DL (kbps)', y='Handset Type', data=handset_stats)
    plt.title('Distribution of Average Throughput (DL) per Handset Type')
    plt.xlabel('Average Throughput DL (kbps)')
    plt.ylabel('Handset Type')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Avg Bearer TP UL (kbps)', y='Handset Type', data=handset_stats)
    plt.title('Distribution of Average Throughput (UL) per Handset Type')
    plt.xlabel('Average Throughput UL (kbps)')
    plt.ylabel('Handset Type')
    plt.show()

    # Plot the average TCP retransmission per handset type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='TCP DL Retrans. Vol (Bytes)', y='Handset Type', data=handset_stats)
    plt.title('Average TCP DL Retransmission Volume per Handset Type')
    plt.xlabel('TCP DL Retransmission Volume (Bytes)')
    plt.ylabel('Handset Type')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='TCP UL Retrans. Vol (Bytes)', y='Handset Type', data=handset_stats)
    plt.title('Average TCP UL Retransmission Volume per Handset Type')
    plt.xlabel('TCP UL Retransmission Volume (Bytes)')
    plt.ylabel('Handset Type')
    plt.show()

    return handset_stats




def perform_kmeans_clustering(df, k=3):
    """Perform k-means clustering on experience metrics to segment users into groups and describe each cluster."""
    
    # Select the relevant experience metrics for clustering
    features = df[['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 
                   'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 
                   'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']]
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Analyze the clusters
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_descriptions = pd.DataFrame(cluster_centers, columns=features.columns)
    cluster_counts = df['Cluster'].value_counts().sort_index()

    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cluster', data=df)
    plt.title('User Distribution Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Users')
    plt.show()

    return cluster_descriptions, cluster_counts


