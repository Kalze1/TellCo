import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def assign_engagement_experience_scores(df, engagement_metrics, experience_metrics, k=3):
    """Assigns engagement and experience scores to each user."""
    
    # Select and standardize the engagement and experience metrics
    engagement_features = df[engagement_metrics]
    experience_features = df[experience_metrics]
    
    scaler_engagement = StandardScaler()
    scaled_engagement_features = scaler_engagement.fit_transform(engagement_features)
    
    scaler_experience = StandardScaler()
    scaled_experience_features = scaler_experience.fit_transform(experience_features)
    
    # Apply k-means clustering for engagement and experience
    kmeans_engagement = KMeans(n_clusters=k, random_state=42)
    kmeans_experience = KMeans(n_clusters=k, random_state=42)
    
    df['Engagement Cluster'] = kmeans_engagement.fit_predict(scaled_engagement_features)
    df['Experience Cluster'] = kmeans_experience.fit_predict(scaled_experience_features)
    
    # Calculate centroids of clusters for engagement and experience
    engagement_centroids = kmeans_engagement.cluster_centers_
    experience_centroids = kmeans_experience.cluster_centers_
    
    # Determine the less engaged cluster (based on centroid position, assume cluster 0 is less engaged)
    less_engaged_centroid = engagement_centroids[0]
    
    # Determine the worst experience cluster (based on centroid position, assume cluster 0 is worst experience)
    worst_experience_centroid = experience_centroids[0]
    
    # Calculate Euclidean distances for engagement and experience scores
    df['Engagement Score'] = euclidean_distances(scaled_engagement_features, [less_engaged_centroid]).flatten()
    df['Experience Score'] = euclidean_distances(scaled_experience_features, [worst_experience_centroid]).flatten()
    
    return df[['MSISDN/Number', 'Engagement Score', 'Experience Score']]



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



def compute_satisfaction_score(df):
    """Compute the satisfaction score as the average of engagement and experience scores, then report top 10 satisfied customers."""
    
    # Calculate the satisfaction score
    df['Satisfaction Score'] = df[['Engagement Score', 'Experience Score']].mean(axis=1)
    
    # Sort customers by satisfaction score in ascending order (since lower scores indicate higher satisfaction)
    top_satisfied_customers = df.sort_values(by='Satisfaction Score').head(10)
    
    return top_satisfied_customers[['MSISDN/Number', 'Satisfaction Score']]



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






def run_kmeans_on_scores(df, k=2):
    """Run k-means clustering on the engagement and experience scores."""
    
    # Select the relevant scores
    X = df[['Engagement Score', 'Experience Score']]
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Score Cluster'] = kmeans.fit_predict(X)
    
    return df, kmeans





def aggregate_scores_per_cluster(df):
    """Aggregate the average satisfaction and experience score per cluster."""
    
    cluster_aggregation = df.groupby('Score Cluster').agg({
        'Satisfaction Score': 'mean',
        'Experience Score': 'mean'
    }).reset_index()
    
    return cluster_aggregation






def export_to_mysql(df, table_name, user, password, host, database):
    """Export the DataFrame to a MySQL database."""
    
    # Create the connection string
    connection_string = f'mysql+mysqlconnector://{user}:{password}@{host}/{database}'
    
    # Create the database engine
    engine = create_engine(connection_string)
    
    # Export the DataFrame to MySQL
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    
    return f"Data exported to MySQL table {table_name}."



