import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def aggregate_user_data(df):
   
    user_aggregates = df.groupby('IMSI').agg(
        num_xdr_sessions=('Bearer Id', 'count'),
        total_session_duration=('Dur. (ms)', 'sum'),
        total_download_data=('Total DL (Bytes)', 'sum'),
        total_upload_data=('Total UL (Bytes)', 'sum'),
        
        # Sum of data per application
        social_media_dl=('Social Media DL (Bytes)', 'sum'),
        social_media_ul=('Social Media UL (Bytes)', 'sum'),
        google_dl=('Google DL (Bytes)', 'sum'),
        google_ul=('Google UL (Bytes)', 'sum'),
        youtube_dl=('Youtube DL (Bytes)', 'sum'),
        youtube_ul=('Youtube UL (Bytes)', 'sum'),
        netflix_dl=('Netflix DL (Bytes)', 'sum'),
        netflix_ul=('Netflix UL (Bytes)', 'sum'),
        gaming_dl=('Gaming DL (Bytes)', 'sum'),
        gaming_ul=('Gaming UL (Bytes)', 'sum'),
        other_dl=('Other DL (Bytes)', 'sum'),
        other_ul=('Other UL (Bytes)', 'sum')
    ).reset_index()
    
    return user_aggregates


# Function to load and inspect data
def load_and_inspect_data(df):
    print(df.info())
    print(df.describe(include='all'))
    return df


# Function to segment users into decile classes based on total session duration
def segment_users_by_duration(df):
    # Calculate total session duration and total data
    df['Total_Session_Duration'] = df['Dur. (ms)']
    df['Total_Data'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    
    # Segment users into decile classes based on session duration
    df['Decile_Class'] = pd.qcut(df['Total_Session_Duration'], 10, labels=False, duplicates='drop')
    
    # Compute total data per decile class
    decile_data = df.groupby('Decile_Class')['Total_Data'].sum().sort_values(ascending=False)
    
    return df, decile_data


# Function to compute basic metrics for quantitative variables
def compute_basic_metrics(df):
    # List of columns that should be numeric
    numeric_columns = [
        'Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)',
        'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
        'Google DL (Bytes)', 'Google UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
        'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
        'Other DL (Bytes)', 'Other UL (Bytes)'
    ]
    
    # Convert columns to numeric, forcing errors to NaN
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Now compute mean, median, std_dev
    mean_values = df[numeric_columns].mean()
    median_values = df[numeric_columns].median()
    std_dev = df[numeric_columns].std()

    print("Mean Values:\n", mean_values)
    print("Median Values:\n", median_values)
    print("Standard Deviation:\n", std_dev)

    # Dispersion parameters
    dispersion_params = df[numeric_columns].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    print(dispersion_params)
    
    return mean_values, median_values, std_dev, dispersion_params


# Function to perform graphical univariate analysis
def univariate_analysis(df, variable):
    # Histogram
    plt.figure(figsize=(10,6))
    sns.histplot(df[variable], bins=20, kde=True)
    plt.title(f"Distribution of {variable}")
    plt.show()

    # Boxplot
    plt.figure(figsize=(10,6))
    sns.boxplot(x=df[variable])
    plt.title(f"Boxplot of {variable}")
    plt.show()

# Function to perform bivariate analysis
def bivariate_analysis(df, app_columns):
    for app in app_columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=df[app], y=df['Total_Data'])
        plt.title(f"Relationship between {app} and Total Data")
        plt.xlabel(app)
        plt.ylabel("Total Data (DL + UL)")
        plt.show()

# Function to compute correlation matrix
def correlation_analysis(df, app_columns):
    correlation_matrix = df[app_columns].corr()
    print(correlation_matrix)
    
    # Heatmap of correlation matrix
    plt.figure(figsize=(10,6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix for Application Data")
    plt.show()

    return correlation_matrix

# Function to perform PCA for dimensionality reduction
def pca_analysis(df, app_columns):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[app_columns])
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Explained variance ratio
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    
    # Visualize PCA components
    plt.figure(figsize=(8,6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title('PCA on Application Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    return pca_result
