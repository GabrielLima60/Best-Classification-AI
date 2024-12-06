import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder

class PrepareData:

    def __init__(self, dataframe):
        if 'id' in dataframe.columns:
            dataframe.drop('id', axis=1, inplace=True)
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        for column in dataframe.columns:
            dataframe[column] = dataframe[column].apply(lambda x: pd.to_numeric(x, errors='coerce')\
                                                        if isinstance(x, str) and any(char.isdigit() for char in x) else x)

        dataframe = dataframe.dropna()

        #dataframe = self.remove_outliers(dataframe) 

        self.x = dataframe.iloc[:, :-1]
        self.x = self.identify_classification_columns_and_get_dummies(self.x)

        self.y = dataframe.iloc[:, -1]
        label_encoder = LabelEncoder()
        self.y = pd.Series(label_encoder.fit_transform(self.y))

        dataframe = pd.concat([self.x, self.y.rename('target')], axis=1)
        dataframe = dataframe.dropna()

        self.save_cleaned_data(dataframe)
    
    def remove_outliers(self, dataframe, threshold=3):

        numerical_df = dataframe.select_dtypes(include=['number'])

        if numerical_df.empty:
            return dataframe

        # Calculate the Q1 (25th percentile) and Q3 (75th percentile) for each column
        Q1 = numerical_df.quantile(0.25)
        Q3 = numerical_df.quantile(0.75)
        
        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1
        
        # Define lower and upper bounds for outlier detection
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Identify outliers
        outliers = (numerical_df < lower_bound) | (numerical_df > upper_bound)
        
        # Remove rows with outliers across any feature column
        df_cleaned = dataframe[~outliers.any(axis=1)]

        return df_cleaned

    def identify_classification_columns_and_get_dummies (self, dataframe):
        potential_categorical_columns = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtype in [int, object, str]]
        dataframe = pd.get_dummies(dataframe, columns=potential_categorical_columns)

        return dataframe
    
    def save_cleaned_data(self, dataframe):
        file_path = os.path.join("resources", "cleaned_data.csv")
        dataframe.to_csv(file_path, index=False)
  

# MAIN

dataframe = sys.argv[1]
dataframe = pd.read_csv(dataframe)
PrepareData(dataframe)
