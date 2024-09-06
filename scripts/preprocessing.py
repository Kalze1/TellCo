import pandas as pd 
import numpy as np

def replace_missing_with_mean(df, column):
    #replace missing values in specified columns with their mean
    df[column] = df[column].fillna(df[column].mean())
    return df 


def replce_messing_with_median(df, column):
    #replace messing values with their median
    df[column] = df[column].fillna(df[column].median)
    return df


def count_missing_values(df):
    #count the missing values and return the percentage of the messing values 
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    return missing_count , missing_percentage

def replace_outliers_with_quartile(df, column, quartile=0.95):
    for col in column:
        # Calculate the specified quartile
        percentile = np.quantile(df[col], quartile)

        # Identify outliers
        outliers = df[col] > percentile

        # Replace outliers with the quartile value
        df.loc[outliers, col] = percentile

    return df


def count_outliers(df,column, quartile = 0.95):
    table = []
    for col in column:

        outlier_count = (df[col] > np.quantile(df[col], quartile)).sum()
        outlier_percentage = (outlier_count / len(df)) * 100

        table.append([col, outlier_count, outlier_percentage])

        

    return pd.DataFrame(table, columns=['Column Name', 'Number of Outliers', 'Outlier Percentage'])
