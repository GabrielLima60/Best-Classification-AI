import pandas as pd
import os
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler

class PrepareData:

    def __init__(self, dataframe, data_cleaning_methods):
        if 'id' in dataframe.columns:
            dataframe.drop('id', axis=1, inplace=True)
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        for column in dataframe.columns:
            dataframe[column] = dataframe[column].apply(lambda x: pd.to_numeric(x, errors='coerce')\
                                                        if isinstance(x, str) and any(char.isdigit() for char in x) else x)


        dataframe = dataframe.dropna()

        self.x = dataframe.iloc[:, :-1]

        if 'Normalize' in data_cleaning_methods:
            numeric_cols = self.x.select_dtypes(include=['float64', 'int64']).columns
            scaler = StandardScaler()
            self.x[numeric_cols] = scaler.fit_transform(self.x[numeric_cols])

        if 'Apply OneHotEncoder' in data_cleaning_methods:
            self.x = self.identify_classification_columns_and_get_dummies(self.x)

        self.y = dataframe.iloc[:, -1]
        label_encoder = LabelEncoder()
        self.y = pd.Series(label_encoder.fit_transform(self.y))

        dataframe = pd.concat([self.x, self.y.rename('target')], axis=1)

        dataframe = dataframe.dropna()

        self.save_cleaned_data(dataframe)

    def identify_classification_columns_and_get_dummies (self, dataframe):
        potential_categorical_columns = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtype in [int, object, str]]
        if len(potential_categorical_columns) > 0:
            dataframe = pd.get_dummies(dataframe, columns=potential_categorical_columns)

        return dataframe
    
    def save_cleaned_data(self, dataframe):
        file_path = os.path.join("resources", "cleaned_data.csv")
        dataframe.to_csv(file_path, index=False)
  

# MAIN

dataframe = sys.argv[1]
data_cleaning_methods = sys.argv[2]
dataframe = pd.read_csv(dataframe)
PrepareData(dataframe, data_cleaning_methods)
