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



import pandas as pd

def compute_satisfaction_score(df):
    """Compute the satisfaction score as the average of engagement and experience scores, then report top 10 satisfied customers."""
    
    # Calculate the satisfaction score
    df['Satisfaction Score'] = df[['Engagement Score', 'Experience Score']].mean(axis=1)
    
    # Sort customers by satisfaction score in ascending order (since lower scores indicate higher satisfaction)
    top_satisfied_customers = df.sort_values(by='Satisfaction Score').head(10)
    
    return top_satisfied_customers[['MSISDN/Number', 'Satisfaction Score']]



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def build_regression_model(df, features, target):
    """Build and evaluate a regression model to predict the satisfaction score."""
    
    # Selecting features and target variable
    X = df[features]
    y = df[target]
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the regression model (Linear Regression in this case)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Plotting the predicted vs actual satisfaction scores
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.title('Predicted vs Actual Satisfaction Scores')
    plt.xlabel('Actual Satisfaction Score')
    plt.ylabel('Predicted Satisfaction Score')
    plt.show()
    
    # Returning the model and evaluation metrics
    return model, mse, r2

# Example usage:
# features = ['Engagement Score', 'Experience Score']  # Add any additional relevant features if needed
# target = 'Satisfaction Score'
# model, mse, r2 = build_regression_model(df, features, target)

# To view the results:
# print(f"Mean Squared Error: {mse}")
# print(f"R-squared: {r2}")


