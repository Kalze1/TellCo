import pandas as pd 
import numpy as np

def replace_missing_with_mean(df, columns):
   
    for column in columns:
        df[column] = df[column].fillna(df[column].mean())
    return df


def replace_missing_with_median(df, columns):
   
    for column in columns:
        df[column] = df[column].fillna(df[column].median())
    return df


def count_missing_values(df):
    #count the missing values and return the percentage of the messing values 
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    return missing_count , missing_percentage

def replace_outliers_with_mean(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

        # Replace outliers with the mean value of the column
        mean_value = df[col].mean()
        df.loc[outliers, col] = mean_value

    return df


def replace_outliers_with_percentile(df, columns, percentile=0.95):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

        # Calculate the specified percentile value (e.g., 95th percentile)
        percentile_value = df[col].quantile(percentile)

        # Replace outliers with the percentile value
        df.loc[outliers, col] = percentile_value

    return df





def count_outliers(df, columns, quartile=0.95):
   
    table = []
    
    for col in columns:
        # Calculate the 1st quartile (25%) and the 3rd quartile (75%)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        
        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1
        
        # Define outliers as values that are 1.5 * IQR above Q3 or below Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers above upper bound and below lower bound
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        # Calculate the percentage of outliers
        outlier_percentage = (outlier_count / len(df)) * 100
        
        # Append the results to the table
        table.append([col, outlier_count, outlier_percentage])
    
    # Return the result as a DataFrame
    return pd.DataFrame(table, columns=['Column Name', 'Number of Outliers', 'Outlier Percentage'])
