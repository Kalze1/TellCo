import streamlit as st
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from load_data import load_df  # Adjust the import path if necessary
from eda_module import aggregate_user_data, bivariate_analysis, compute_basic_metrics, correlation_analysis, pca_analysis, segment_users_by_duration  # Adjust the import path if necessary

# Load the data
df = load_df()


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





# def bivariate_analysis(df, app_columns):
#     for app in app_columns:
#         fig, ax = plt.subplots(figsize=(8, 5))
#         sns.scatterplot(x=df[app], y=df['Total_Data'], ax=ax)
#         ax.set_title(f"Relationship between {app} and Total Data")
#         ax.set_xlabel(app)
#         ax.set_ylabel("Total Data (DL + UL)")
#         st.pyplot(fig)






# Streamlit app layout
st.title('Telco Data Analysis Dashboard')
st.header('Overview')
st.write('This dashboard presents the analysis of Telco user data.')

# # Display the entire dataset
# st.subheader('Entire Dataset')
# st.write(df)

# # Display the aggregated data
# st.subheader('Aggregated User Data')
# st.write(user_aggregates)

# Visualization: Number of xDR Sessions per User
# st.subheader('Number of xDR Sessions per User')
# fig, ax = plt.subplots()
# ax.bar(user_aggregates['IMSI'], user_aggregates['num_xdr_sessions'])
# ax.set_xlabel('User (IMSI)')
# ax.set_ylabel('Number of xDR Sessions')
# ax.set_title('Number of xDR Sessions per User')
# st.pyplot(fig)

# # Visualization: Total Session Duration per User
# st.subheader('Total Session Duration per User')
# fig, ax = plt.subplots()
# ax.bar(user_aggregates['IMSI'], user_aggregates['total_session_duration'])
# ax.set_xlabel('User (IMSI)')
# ax.set_ylabel('Total Session Duration (ms)')
# ax.set_title('Total Session Duration per User')
# st.pyplot(fig)


# # Visualization: Total Download Data per User
# st.subheader('Total Download Data per User')
# fig, ax = plt.subplots()
# ax.bar(user_aggregates['IMSI'], user_aggregates['total_download_data'])
# ax.set_xlabel('User (IMSI)')
# ax.set_ylabel('Total Download Data (Bytes)')
# ax.set_title('Total Download Data per User')
# st.pyplot(fig)

# # Visualization: Total Upload Data per User
# st.subheader('Total Upload Data per User')
# fig, ax = plt.subplots()
# ax.bar(user_aggregates['IMSI'], user_aggregates['total_upload_data'])
# ax.set_xlabel('User (IMSI)')
# ax.set_ylabel('Total Upload Data (Bytes)')
# ax.set_title('Total Upload Data per User')
# st.pyplot(fig)

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



# Bivariate Analysis
# Bivariate Analysis
# st.subheader('Bivariate Analysis')
# app_columns = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 
#                'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
#                'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)']
# bivariate_analysis(df, app_columns)


# visualize_correlation(df, app_columns)

# visualize_pca(df, app_columns)