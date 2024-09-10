from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from customer_satisfaction_analysis import aggregate_scores_per_cluster, assign_engagement_experience_scores, run_kmeans_on_scores
from experiance_segmentation import aggregate_per_customer, analyze_throughput_tcp_retransmission, build_regression_model, compute_satisfaction_score, compute_top_bottom_frequent, perform_kmeans_clustering
from user_engagement import aggregate_engagement_metrics, aggregate_traffic_per_app, compute_cluster_stats, elbow_method, normalize_and_cluster, plot_top_3_apps, silhouette_analysis
# from load_data import load_df  
from eda_module import aggregate_user_data, bivariate_analysis, compute_basic_metrics, correlation_analysis, pca_analysis, segment_users_by_duration  

# Load the data
df = load_df()
df= df.head()


#Numeric columns 
numeric_columns = [
        'Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)',
        'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
        'Google DL (Bytes)', 'Google UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
        'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
        'Other DL (Bytes)', 'Other UL (Bytes)'
    ]


app_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
               'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']

# Aggregate the data
user_aggregates = aggregate_user_data(df)


def visualize_decile_data(decile_data):
    # Create a bar chart
    fig, ax = plt.subplots()
    ax.bar(decile_data.index, decile_data.values)
    ax.set_xlabel('Decile Class')
    ax.set_ylabel('Total Data (Bytes)')
    ax.set_title('Total Data (DL+UL) per Decile Class')
    
    # Display the chart in Streamlit
    st.pyplot(fig)


def visualize_xdr_sessions(user_aggregates):
    st.subheader('Number of xDR Sessions per User')
    fig, ax = plt.subplots()
    ax.bar(user_aggregates['IMSI'], user_aggregates['num_xdr_sessions'])
    ax.set_xlabel('User (IMSI)')
    ax.set_ylabel('Number of xDR Sessions')
    ax.set_title('Number of xDR Sessions per User')
    st.pyplot(fig)

def visualize_total_session_duration(user_aggregates):
    st.subheader('Total Session Duration per User')
    fig, ax = plt.subplots()
    ax.bar(user_aggregates['IMSI'], user_aggregates['total_session_duration'])
    ax.set_xlabel('User (IMSI)')
    ax.set_ylabel('Total Session Duration (ms)')
    ax.set_title('Total Session Duration per User')
    st.pyplot(fig)

def visualize_total_download_data(user_aggregates):
    st.subheader('Total Download Data per User')
    fig, ax = plt.subplots()
    ax.bar(user_aggregates['IMSI'], user_aggregates['total_download_data'])
    ax.set_xlabel('User (IMSI)')
    ax.set_ylabel('Total Download Data (Bytes)')
    ax.set_title('Total Download Data per User')
    st.pyplot(fig)

def visualize_total_upload_data(user_aggregates):
    st.subheader('Total Upload Data per User')
    fig, ax = plt.subplots()
    ax.bar(user_aggregates['IMSI'], user_aggregates['total_upload_data'])
    ax.set_xlabel('User (IMSI)')
    ax.set_ylabel('Total Upload Data (Bytes)')
    ax.set_title('Total Upload Data per User')
    st.pyplot(fig)



def visualize_basic_metrics(mean_values, median_values, std_dev, dispersion_params):
    # Selection box for metrics
    metric = st.selectbox('Select a metric to display', ['Mean', 'Median', 'Standard Deviation'])
    
    if metric == 'Mean':
        st.subheader('Mean Values')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Mean Values:', mean_values)
        with col2:
            fig, ax = plt.subplots()
            mean_values.plot(kind='bar', ax=ax)
            ax.set_ylabel('Mean')
            ax.set_title('Mean Values for Numeric Columns')
            st.pyplot(fig)
    
    elif metric == 'Median':
        st.subheader('Median Values')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Median Values:', median_values)
        with col2:
            fig, ax = plt.subplots()
            median_values.plot(kind='bar', ax=ax)
            ax.set_ylabel('Median')
            ax.set_title('Median Values for Numeric Columns')
            st.pyplot(fig)
    
    elif metric == 'Standard Deviation':
        st.subheader('Standard Deviation')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Standard Deviation:', std_dev)
        with col2:
            fig, ax = plt.subplots()
            std_dev.plot(kind='bar', ax=ax)
            ax.set_ylabel('Standard Deviation')
            ax.set_title('Standard Deviation for Numeric Columns')
            st.pyplot(fig)
    
    

def dispersion_parameters():
    st.subheader('Dispersion Parameters')
    fig,ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df[numeric_columns], ax=ax)
    ax.set_title('Dispersion Parameters for Numeric Columns')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90) 
    st.pyplot(fig)


def visualize_correlation(df, app_columns):
    st.write("## Correlation Matrix for Application Data")
    
    # Compute the correlation matrix
    correlation_matrix = correlation_analysis(df, app_columns)
    
    # Display the correlation matrix as a heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def visualize_pca(df, app_columns):
    st.write("## PCA for Dimensionality Reduction")
    
    # Perform PCA
    pca_result = pca_analysis(df, app_columns)
    
    # Visualize PCA components
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pca_result[:, 0], pca_result[:, 1])
    ax.set_title('PCA on Application Data')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    st.pyplot(fig)


def display_user_engagement_metrics(df):
    st.write("## User Engagement Metrics")
    
    # Aggregate user engagement metrics
    user_metrics = aggregate_engagement_metrics(df)
    
    # Display the aggregated metrics
    st.write(user_metrics)
    
    # Optionally, visualize some of the metrics
    st.write("### Total Traffic per User")
    st.bar_chart(user_metrics[['MSISDN/Number', 'total_traffic']].set_index('MSISDN/Number'))



def visualize_clusters(df, k=3):
    st.write("## K-Means Clustering of Engagement Metrics")
    
    # Normalize and apply k-means clustering
    clustered_df, kmeans = normalize_and_cluster(df, k)
    
    # Display the clustered data
    st.write(clustered_df)
    
    # Visualize the clusters
    st.write("### Cluster Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(clustered_df['total_traffic'], clustered_df['total_duration'], c=clustered_df['cluster'], cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_title('K-Means Clustering of Engagement Metrics')
    ax.set_xlabel('Total Traffic')
    ax.set_ylabel('Total Duration')
    st.pyplot(fig)

def display_cluster_stats(df):
    st.write("## Cluster Statistics")
    
    # Compute cluster statistics
    cluster_stats = compute_cluster_stats(df)
    
    # Display the cluster statistics
    st.write(cluster_stats)
    
    # Optionally, visualize some of the statistics
    st.write("### Average Sessions per Cluster")
    st.bar_chart(cluster_stats[['cluster', 'avg_sessions']].set_index('cluster'))
    
    st.write("### Average Duration per Cluster")
    st.bar_chart(cluster_stats[['cluster', 'avg_duration']].set_index('cluster'))
    
    st.write("### Average Traffic per Cluster")
    st.bar_chart(cluster_stats[['cluster', 'avg_traffic']].set_index('cluster'))






def bivariate_analysis(df, app_columns):
    for app in app_columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=df[app], y=df['Total_Data'], ax=ax)
        ax.set_title(f"Relationship between {app} and Total Data")
        ax.set_xlabel(app)
        ax.set_ylabel("Total Data (DL + UL)")
        st.pyplot(fig)



def visualize_app_traffic(df):
    st.write("## Traffic per Application")
    
    # Aggregate traffic per application
    app_traffic = aggregate_traffic_per_app(df)
    
    # Display the aggregated traffic
    st.write(app_traffic)
    
    # Visualize the traffic per application
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(app_traffic['Application'], app_traffic['Total Traffic (Bytes)'])
    ax.set_title('Total Traffic per Application')
    ax.set_xlabel('Application')
    ax.set_ylabel('Total Traffic (Bytes)')
    plt.xticks(rotation=45)
    st.pyplot(fig)


def visualize_top_3_apps(df):
    app_traffic = aggregate_traffic_per_app(df)
    st.write("## Top 3 Most Used Applications by Traffic")
    
    # Plot the top 3 most used applications
    top_3_apps = plot_top_3_apps(app_traffic)
    
    # Display the top 3 applications
    st.write(top_3_apps)
    
    # Visualize the top 3 applications
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Application', y='Total Traffic (Bytes)', data=top_3_apps, ax=ax)
    ax.set_title('Top 3 Most Used Applications by Traffic')
    ax.set_xlabel('Application')
    ax.set_ylabel('Total Traffic (Bytes)')
    st.pyplot(fig)


def visualize_elbow_method(df):
    st.write("## Elbow Method for Optimal k")
    
    # Use the elbow method
    inertia = elbow_method(df)
    
    # Visualize the elbow curve
    fig, ax = plt.subplots(figsize=(8, 6))
    k_range = range(1, 11)
    ax.plot(k_range, inertia, 'bo-')
    ax.set_title('Elbow Method For Optimal k')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    st.pyplot(fig)


def visualize_silhouette_score(df, k):
    st.write(f"## Silhouette Score for k={k}")
    
    # Compute the silhouette score
    score = silhouette_analysis(df, k)
    
    # Display the silhouette score
    st.write(f"Silhouette Score: {score}")


def visualize_customer_aggregation(df):
    st.write("## Customer Aggregated Metrics")
    
    # Aggregate data per customer
    customer_data = aggregate_per_customer(df)
    
    # Display the aggregated data
    st.write(customer_data)
    
    # Visualize some of the aggregated metrics
    st.write("### Average TCP DL Retrans (Bytes) per Customer")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Customer', y='Avg TCP DL Retrans (Bytes)', data=customer_data, ax=ax)
    ax.set_title('Average TCP DL Retrans (Bytes) per Customer')
    ax.set_xlabel('Customer')
    ax.set_ylabel('Avg TCP DL Retrans (Bytes)')
    plt.xticks(rotation=90)
    st.pyplot(fig)
    
    st.write("### Average Throughput DL (kbps) per Customer")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Customer', y='Avg Throughput DL (kbps)', data=customer_data, ax=ax)
    ax.set_title('Average Throughput DL (kbps) per Customer')
    ax.set_xlabel('Customer')
    ax.set_ylabel('Avg Throughput DL (kbps)')
    plt.xticks(rotation=90)
    st.pyplot(fig)

def visualize_top_bottom_frequent(df, column, top_n=10):
    st.write(f"## Top, Bottom, and Most Frequent Values for {column}")
    
    # Compute the top, bottom, and most frequent values
    top_values, bottom_values, frequent_values = compute_top_bottom_frequent(df, column, top_n)
    
    # Display the top values
    st.write(f"### Top {top_n} Values")
    st.write(top_values)
    
    # Display the bottom values
    st.write(f"### Bottom {top_n} Values")
    st.write(bottom_values)
    
    # Display the most frequent values
    st.write(f"### Most Frequent {top_n} Values")
    st.write(frequent_values)
    
    # Visualize the top values
    st.write(f"### Top {top_n} Values Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_values.index, y=top_values.values, ax=ax)
    ax.set_title(f'Top {top_n} Values for {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Value')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Visualize the bottom values
    st.write(f"### Bottom {top_n} Values Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=bottom_values.index, y=bottom_values.values, ax=ax)
    ax.set_title(f'Bottom {top_n} Values for {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Value')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Visualize the most frequent values
    st.write(f"### Most Frequent {top_n} Values Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=frequent_values.index, y=frequent_values.values, ax=ax)
    ax.set_title(f'Most Frequent {top_n} Values for {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def visualize_handset_stats(df):
    st.write("## Handset Throughput and TCP Retransmission Analysis")
    
    # Analyze throughput and TCP retransmission
    handset_stats = analyze_throughput_tcp_retransmission(df)
    
    # Display the handset statistics
    st.write(handset_stats)
    
    # Visualize the distribution of average throughput (DL) per handset type
    st.write("### Distribution of Average Throughput (DL) per Handset Type")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Avg Bearer TP DL (kbps)', y='Handset Type', data=handset_stats, ax=ax)
    ax.set_title('Distribution of Average Throughput (DL) per Handset Type')
    ax.set_xlabel('Average Throughput DL (kbps)')
    ax.set_ylabel('Handset Type')
    st.pyplot(fig)
    
    # Visualize the distribution of average throughput (UL) per handset type
    st.write("### Distribution of Average Throughput (UL) per Handset Type")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Avg Bearer TP UL (kbps)', y='Handset Type', data=handset_stats, ax=ax)
    ax.set_title('Distribution of Average Throughput (UL) per Handset Type')
    ax.set_xlabel('Average Throughput UL (kbps)')
    ax.set_ylabel('Handset Type')
    st.pyplot(fig)
    
    # Visualize the average TCP DL retransmission volume per handset type
    st.write("### Average TCP DL Retransmission Volume per Handset Type")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='TCP DL Retrans. Vol (Bytes)', y='Handset Type', data=handset_stats, ax=ax)
    ax.set_title('Average TCP DL Retransmission Volume per Handset Type')
    ax.set_xlabel('TCP DL Retransmission Volume (Bytes)')
    ax.set_ylabel('Handset Type')
    st.pyplot(fig)
    
    # Visualize the average TCP UL retransmission volume per handset type
    st.write("### Average TCP UL Retransmission Volume per Handset Type")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='TCP UL Retrans. Vol (Bytes)', y='Handset Type', data=handset_stats, ax=ax)
    ax.set_title('Average TCP UL Retransmission Volume per Handset Type')
    ax.set_xlabel('TCP UL Retransmission Volume (Bytes)')
    ax.set_ylabel('Handset Type')
    st.pyplot(fig)



def visualize_kmeans_clusters(df, k=3):
    st.write("## K-Means Clustering Analysis")
    
    # Perform k-means clustering
    cluster_descriptions, cluster_counts = perform_kmeans_clustering(df, k)
    
    # Display the cluster descriptions
    st.write("### Cluster Descriptions")
    st.write(cluster_descriptions)
    
    # Display the cluster counts
    st.write("### Cluster Counts")
    st.write(cluster_counts)
    
    # Visualize the user distribution across clusters
    st.write("### User Distribution Across Clusters")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Cluster', data=df, ax=ax)
    ax.set_title('User Distribution Across Clusters')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Users')
    st.pyplot(fig)

def visualize_satisfaction_scores(df):
    st.write("## Top 10 Satisfied Customers")
    
    # Compute the satisfaction scores
    top_satisfied_customers = compute_satisfaction_score(df)
    
    # Display the top 10 satisfied customers
    st.write(top_satisfied_customers)
    
    # Visualize the satisfaction scores
    st.write("### Satisfaction Scores of Top 10 Customers")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='MSISDN/Number', y='Satisfaction Score', data=top_satisfied_customers, ax=ax)
    ax.set_title('Top 10 Satisfied Customers')
    ax.set_xlabel('Customer')
    ax.set_ylabel('Satisfaction Score')
    plt.xticks(rotation=45)
    st.pyplot(fig)



def visualize_regression_model(df, features, target):
    st.write("## Regression Model to Predict Satisfaction Score")
    
    # Build and evaluate the regression model
    model, mse, r2 = build_regression_model(df, features, target)
    
    # Display the evaluation metrics
    st.write(f"### Mean Squared Error: {mse}")
    st.write(f"### R-squared: {r2}")
    
    # Plotting the predicted vs actual satisfaction scores
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_title('Predicted vs Actual Satisfaction Scores')
    ax.set_xlabel('Actual Satisfaction Score')
    ax.set_ylabel('Predicted Satisfaction Score')
    st.pyplot(fig)



def visualize_engagement_experience_scores(df, engagement_metrics, experience_metrics, k=3):
    st.write("## Engagement and Experience Scores")
    
    # Assign engagement and experience scores
    scores_df = assign_engagement_experience_scores(df, engagement_metrics, experience_metrics, k)
    
    # Display the scores
    st.write(scores_df)
    
    # Visualize the engagement scores
    st.write("### Engagement Scores")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(scores_df['Engagement Score'], bins=20, kde=True, ax=ax)
    ax.set_title('Distribution of Engagement Scores')
    ax.set_xlabel('Engagement Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    # Visualize the experience scores
    st.write("### Experience Scores")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(scores_df['Experience Score'], bins=20, kde=True, ax=ax)
    ax.set_title('Distribution of Experience Scores')
    ax.set_xlabel('Experience Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)



def visualize_regression_model(df, features, target):
    st.write("## Regression Model to Predict Satisfaction Score")
    
    # Build and evaluate the regression model
    model, mse, r2 = build_regression_model(df, features, target)
    
    # Display the evaluation metrics
    st.write(f"### Mean Squared Error: {mse}")
    st.write(f"### R-squared: {r2}")
    
    # Plotting the predicted vs actual satisfaction scores
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_title('Predicted vs Actual Satisfaction Scores')
    ax.set_xlabel('Actual Satisfaction Score')
    ax.set_ylabel('Predicted Satisfaction Score')
    st.pyplot(fig)






def visualize_kmeans_clusters(df, k=2):
    st.write("## K-Means Clustering on Engagement and Experience Scores")
    
    # Run k-means clustering
    clustered_df, kmeans = run_kmeans_on_scores(df, k)
    
    # Display the clustered data
    st.write(clustered_df)
    
    # Plot the clusters
    st.write("### Clusters Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Engagement Score', y='Experience Score', hue='Score Cluster', data=clustered_df, palette='viridis', ax=ax)
    ax.set_title('K-Means Clustering')
    ax.set_xlabel('Engagement Score')
    ax.set_ylabel('Experience Score')
    st.pyplot(fig)
    
    # Plot the cluster centers
    st.write("### Cluster Centers")
    centers = kmeans.cluster_centers_
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Engagement Score', y='Experience Score', hue='Score Cluster', data=clustered_df, palette='viridis', ax=ax)
    ax.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', label='Centers')
    ax.set_title('Cluster Centers')
    ax.set_xlabel('Engagement Score')
    ax.set_ylabel('Experience Score')
    ax.legend()
    st.pyplot(fig)




def visualize_aggregated_scores(df):
    st.write("## Aggregated Scores per Cluster")
    
    # Aggregate the scores per cluster
    cluster_aggregation = aggregate_scores_per_cluster(df)
    
    # Display the aggregated scores
    st.write(cluster_aggregation)
    
    # Visualize the aggregated satisfaction scores per cluster
    st.write("### Average Satisfaction Score per Cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Score Cluster', y='Satisfaction Score', data=cluster_aggregation, ax=ax)
    ax.set_title('Average Satisfaction Score per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Average Satisfaction Score')
    st.pyplot(fig)
    
    # Visualize the aggregated experience scores per cluster
    st.write("### Average Experience Score per Cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Score Cluster', y='Experience Score', data=cluster_aggregation, ax=ax)
    ax.set_title('Average Experience Score per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Average Experience Score')
    st.pyplot(fig)






# # Streamlit app layout
# st.title('Telco Data Analysis Dashboard')
# st.header('Overview')
# st.write('This dashboard presents the analysis of Telco user data.')

# # # Display the entire dataset
# st.subheader('Entire Dataset')
# st.write(df)

# # # Display the aggregated data
# st.subheader('Aggregated User Data')
# st.write(user_aggregates)



#  # Visualize the number of xDR sessions per user
# visualize_xdr_sessions(user_aggregates)

# # Visualize the total session duration per user
# visualize_total_session_duration(user_aggregates)

# # Visualize the total download data per user
# visualize_total_download_data(user_aggregates)

# # Visualize the total upload data per user
# visualize_total_upload_data(user_aggregates)




# # Visualization: Application-Specific Data Usage per User
# applications = ['social_media_dl', 'social_media_ul', 'google_dl', 'google_ul', 'youtube_dl', 'youtube_ul', 'netflix_dl', 'netflix_ul', 'gaming_dl', 'gaming_ul', 'other_dl', 'other_ul']
# for app in applications:
#     st.subheader(f'Total {app.replace("_", " ").title()} per User')
#     fig, ax = plt.subplots()
#     ax.bar(user_aggregates['IMSI'], user_aggregates[app])
#     ax.set_xlabel('User (IMSI)')
#     ax.set_ylabel(f'Total {app.replace("_", " ").title()} (Bytes)')
#     ax.set_title(f'Total {app.replace("_", " ").title()} per User')
#     st.pyplot(fig)





# df, decile_data = segment_users_by_duration(df)
# # Visualization: Total Data (DL+UL) per Decile Class
# st.subheader('Total Data (DL+UL) per Decile Class')
# visualize_decile_data(decile_data)

# mean_values, median_values, std_dev, dispersion_params = compute_basic_metrics(df)
# # Display the basic metrics
# st.subheader('Basic Metrics')
# # Visualization: Basic Metrics
# visualize_basic_metrics(mean_values, median_values, std_dev, dispersion_params)

# dispersion_parameters()




# # Bivariate Analysis
# st.subheader('Bivariate Analysis')
# app_columns = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 
#                'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
#                'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)']
# bivariate_analysis(df, app_columns)


# visualize_correlation(df, app_columns)

# visualize_pca(df, app_columns)

# # Display user engagement metrics
# display_user_engagement_metrics(df)

# # Display the clustering results
# visualize_clusters(df, k=3)


# display_cluster_stats(df)


# # Visualize the traffic per application
# visualize_app_traffic(df)


#  # Visualize the top 3 most used applications
# visualize_top_3_apps(df)


#  # Visualize the elbow method
# visualize_elbow_method(df)

# # Visualize the silhouette score for a chosen k
# k = 3 

# # You can change this value based on your analysis
# visualize_silhouette_score(df, k)


# # Visualize the aggregated data per customer
# visualize_customer_aggregation(df)


#  # Define the column you want to analyze
# column = 'your_column_name'  

# # Visualize the top, bottom, and most frequent values
# visualize_top_bottom_frequent(df, column, top_n=10)


#  # Visualize the handset statistics
# visualize_handset_stats(df)



# # Visualize the k-means clusters
# visualize_kmeans_clusters(df, k=3)

#  # Visualize the top 10 satisfied customers
# visualize_satisfaction_scores(df)



# # Define the features and target variable
# features = ['Engagement Score', 'Experience Score']  # Replace with your actual feature columns
# target = 'Satisfaction Score'  # Replace with your actual target column
    
# # Visualize the regression model results
# visualize_regression_model(df, features, target)


#  # Define the engagement and experience metrics
# engagement_metrics = ['session_count', 'total_duration', 'total_traffic']  # Replace with your actual engagement metrics
# experience_metrics = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']  # Replace with your actual experience metrics

# # # Visualize the engagement and experience scores
# visualize_engagement_experience_scores(df, engagement_metrics, experience_metrics, k=3)

#  # Define the features and target variable
# features = ['Engagement Score', 'Experience Score']  # Replace with your actual feature columns
# target = 'Satisfaction Score'  # Replace with your actual target column

# # Visualize the regression model results
# visualize_regression_model(df, features, target)

#  # Visualize the k-means clustering results
# visualize_kmeans_clusters(df, k=2)

#  # Visualize the aggregated scores per cluster
# visualize_aggregated_scores(df)
