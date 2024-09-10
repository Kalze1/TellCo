import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import replace_missing_with_mean, replace_outliers_with_mean
from customer_satisfaction_analysis import compute_satisfaction_score
from reserv import dispersion_parameters, display_user_engagement_metrics, visualize_app_traffic, visualize_basic_metrics, visualize_correlation, visualize_decile_data, visualize_decliced_data, visualize_elbow_method, visualize_handset_stats, visualize_kmeans_clusters, visualize_pca, visualize_regression_model, visualize_silhouette_score, visualize_top_3_apps
import plotly.express as px 
import plotly.graph_objects as go 
from load_data import load_df  
from eda_module import aggregate_user_data, bivariate_analysis, compute_basic_metrics, create_and_normalize_columns,  segment_users_by_duration  

# Load the data
# URL of the CSV file on Google Drive
csv_url = 'https://drive.google.com/file/d/1WKoMNNlQneudk5JY0JEvtRkCVYIUQYZK/view?usp=sharing'

# Load the CSV file
df = pd.read_csv(csv_url)
# df = load_df()
# df = df.head()

#defining the numeric columns
columns = ['Dur. (ms)', 'Avg RTT DL (ms)',
       'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
       'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
       'DL TP < 50 Kbps (%)', '50 Kbps < DL TP < 250 Kbps (%)',
       '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)',
       'UL TP < 10 Kbps (%)', '10 Kbps < UL TP < 50 Kbps (%)',
       '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)',
       'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Activity Duration DL (ms)',
       'Activity Duration UL (ms)', 'Dur. (ms).1', 'Nb of sec with 125000B < Vol DL',
       'Nb of sec with 1250B < Vol UL < 6250B',
       'Nb of sec with 31250B < Vol DL < 125000B',
       'Nb of sec with 37500B < Vol UL',
       'Nb of sec with 6250B < Vol DL < 31250B',
       'Nb of sec with 6250B < Vol UL < 37500B',
       'Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B',
       'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
       'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
       'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
       'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)',
       'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)',
       'Total UL (Bytes)', 'Total DL (Bytes)']


#preprocessing the data
df = replace_missing_with_mean(df,columns)
df = replace_outliers_with_mean(df, columns)

# Numeric columns
numeric_columns = [
    'Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)',
    'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
    'Google DL (Bytes)', 'Google UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
    'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
    'Other DL (Bytes)', 'Other UL (Bytes)'
]

app_columns = [
    'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
    'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)'
]

# Aggregate the data
user_aggregates = aggregate_user_data(df)

# Set Streamlit theme
sns.set_theme(style="whitegrid")
# st.set_page_config(layout="wide")

# Create Sidebar


st.sidebar.title("Select Analysis Type")
selected_type = st.sidebar.selectbox("Choose Type ", ["User Overview Analysis","User Engagement Analysis", "Experience anysis", "Satisfaction analysis","Conclusion"])

if selected_type == "User Overview Analysis":

    st.sidebar.title("User Overview Analysis")
    selected_analysis = st.sidebar.selectbox("Choose Analysis", [
        "Overview","Aggregated Data", "User Metrics", "Application Usage", "Decile Class"])
    st.title('Unleash the Potential of TellCo.')

    # Display the entire dataset (optional)
    if selected_analysis == "Overview":
        st.header('Overview')
        st.write("Explore our interactive dashboard to gain valuable insights into customer behavior, optimize product offerings, enhance network performance, improve customer retention, and leverage emerging technologies. Make data-driven decisions to unlock TellCo's full potential.")
        st.subheader('Entire Dataset')
        st.write(df)

    # Visualization: Total Data (DL+UL) per Decile Class
    if selected_analysis == "Decile Class":
        df, decile_data = segment_users_by_duration(df)
        st.subheader('Total Data (DL+UL) per Decile Class')
        visualize_decile_data(decile_data)

    # Visualization: Basic Metrics
    if selected_analysis == "Decile Class":
        mean_values, median_values, std_dev, dispersion_params = compute_basic_metrics(df)
        st.subheader('Basic Metrics')
        visualize_basic_metrics(mean_values, median_values, std_dev, dispersion_params)
        dispersion_parameters()

    # # Display the aggregated data
    if selected_analysis == "Aggregated Data":
        st.subheader('Aggregated User Data')
        st.write(user_aggregates)
        



    # User Metrics Visualization
    if selected_analysis == "User Metrics":
        st.header("User Metrics Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('xDR Sessions per User')
            fig = px.bar(user_aggregates, x='IMSI', y='num_xdr_sessions', title='Number of xDR Sessions per User')
            st.plotly_chart(fig)

            st.subheader('Total Session Duration per User')
            fig = px.bar(user_aggregates, x='IMSI', y='total_session_duration', title='Total Session Duration (ms) per User')
            st.plotly_chart(fig)

        with col2:
            st.subheader('Total Download Data per User')
            fig = px.bar(user_aggregates, x='IMSI', y='total_download_data', title='Total Download Data (Bytes) per User')
            st.plotly_chart(fig)

            st.subheader('Total Upload Data per User')
            fig = px.bar(user_aggregates, x='IMSI', y='total_upload_data', title='Total Upload Data (Bytes) per User')
            st.plotly_chart(fig)

    # Application Usage Visualization
    if selected_analysis == "Application Usage":
        st.header("Application-Specific Data Usage per User")

        applications = [
            'social_media_dl', 'social_media_ul', 'google_dl', 'google_ul', 
            'youtube_dl', 'youtube_ul', 'netflix_dl', 'netflix_ul', 
            'gaming_dl', 'gaming_ul', 'other_dl', 'other_ul'
        ]
        
        for app in applications:
            st.subheader(f'Total {app.replace("_", " ").title()} per User')
            fig = px.bar(user_aggregates, x='IMSI', y=app, title=f'Total {app.replace("_", " ").title()} per User')
            st.plotly_chart(fig)


# User engagement analysis

if selected_type == "User Engagement Analysis":

    st.sidebar.title("User Engagement Analysis")
    selected_analysis = st.sidebar.selectbox("Choose Analysis", [
        "Bivariate Analysis", "correlation between app", "principal components analysis (PCA)", "user engagement metrics", "Traffic per application", "Top 3 most used applications"])
    st.title('Telco Data Analysis Dashboard')
    st.write('This dashboard presents the analysis of Telco user data.')

    # Bivariate Analysis
    if selected_analysis == "Bivariate Analysis":
        st.subheader('Bivariate Analysis')
        app_columns = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 
               'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
               'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)']
        decliced_data = bivariate_analysis(df, app_columns)
        visualize_decliced_data(decliced_data)

    # Visualize the correlation between app usage metrics
    if selected_analysis == "correlation between app":
        st.header("correlation between app usage metrics")
        visualize_correlation(df, app_columns)

    # Visualize the principal components analysis (PCA) for app usage
    if selected_analysis == "principal components analysis (PCA)":
        st.header("principal components analysis (PCA) for app usage")
        visualize_pca(df, app_columns)


    # Display user engagement metrics
    if selected_analysis == "user engagement metrics":
        st.header("user engagement metrics")
        display_user_engagement_metrics(df)


    # Visualize the traffic per application
    if selected_analysis == "Traffic per application":
        st.header("Traffic per application")
        visualize_app_traffic(df)


    # Visualize the top 3 most used applications
    if selected_analysis == "Top 3 most used applications":
        st.header("Top 3 most used applications")
        visualize_top_3_apps(df)




# Experience anysis

if selected_type == "Experience anysis":

    st.sidebar.title("User Experience anysis")
    selected_analysis = st.sidebar.selectbox("Choose Analysis", [
        "Elbow method", "Silhouette score for a chosen k", "k-means clusters for user experience", "The handset statistics"])
    st.title('Telco Data Analysis Dashboard')
    st.write('This dashboard presents the analysis of Telco user data.')

    # Visualize the elbow method (for KMeans clustering)
    if selected_analysis == "Elbow method":
        st.header('Elbow method')
        df = create_and_normalize_columns(df)
        visualize_elbow_method(df)

    # Visualize the silhouette score for a chosen k
    if selected_analysis == "Silhouette score for a chosen k":
        st.header('Silhouette score for a chosen k')
        df = create_and_normalize_columns(df)

        k = 3  # can change this value based on your analysis
        visualize_silhouette_score(df, k)


    # Visualize the k-means clusters for user experience
    if selected_analysis == "k-means clusters for user experience":
        st.header('k-means clusters for user experience')
        df = create_and_normalize_columns(df)

        visualize_kmeans_clusters(df, k=3)

    # Visualize the handset statistics
    if selected_analysis == "The handset statistics":
        st.header('The handset statistics')
        visualize_handset_stats(df)
        

#Satisfaction analysis


if selected_type == "Satisfaction analysis":
    st.header("Customer Satisfaction Analysis")

    st.subheader("Top 10 Satisfied Customers")
    top_satisfied_customers = compute_satisfaction_score(df)
    fig = px.bar(top_satisfied_customers.head(10), x='MSISDN/Number', y='Satisfaction Score', title='Top 10 Satisfied Customers')
    st.plotly_chart(fig)

    st.subheader("Satisfaction Score Prediction")
    features = st.multiselect("Select Features for Regression", ['Engagement Score', 'Experience Score'])
    target = st.selectbox("Select Target Variable", ['Satisfaction Score'])

    if st.button("Run Regression Model"):
        visualize_regression_model(df, features, target)

if selected_type == "Conclusion":
    st.header("Conclusion")
    
# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed by Kaleab Zegeye (https://github.com/Kalze1)")
