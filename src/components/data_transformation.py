import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    

    def get_data_transformer_object(self):
        logging.info("Entered into get tranformation object function")
        try:
            # Define categorical and numeric features
            categorical_columns = ['DAY_OF_WEEK', 'DISTRICT', 'UCR_PART', 'STREET']
            numerical_columns = ['LATITUDE', 'LONGITUDE', 'REPORTING_AREA_STR', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'SHOOTING']

            # Define preprocessing steps for categorical features
            cat_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal_encoder", OrdinalEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Define preprocessing steps for numeric features
            # Define preprocessing steps for numeric features
            num_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('standard_scaler', StandardScaler())
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Define column transformer to apply preprocessing steps to each feature type
            preprocessor = ColumnTransformer(
                transformers=[
                ('num_pipeline', num_transformer, numerical_columns),
                ('cat_pipeline', cat_transformer, categorical_columns)
                ]
            )


            logging.info("Completed the get tranformation object function")

            return preprocessor
        

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, data_path):
        logging.info("Entered the data transformation method or component")
        try:
            # Read the data
            data = pd.read_csv(data_path, parse_dates=['OCCURRED_ON_DATE'], low_memory=False)

            # drop INCIDENT_NUMBER feature
            data.drop(columns=['INCIDENT_NUMBER'], axis=1, inplace=True)

            # Imputing SHOOTING columns as it contains more NAN values
            data['SHOOTING'] = np.where(data['SHOOTING'] == 'Y', 1, 0)

            # Extract date feature
            data['OCCURRED_ON_DATE'] = pd.to_datetime(data['OCCURRED_ON_DATE'])
            data['YEAR'] = data['OCCURRED_ON_DATE'].dt.year
            data['MONTH'] = data['OCCURRED_ON_DATE'].dt.month
            data['DAY'] = data['OCCURRED_ON_DATE'].dt.day
            data['HOUR'] = data['OCCURRED_ON_DATE'].dt.hour

            # Drop the original date column
            data.drop('OCCURRED_ON_DATE', axis=1, inplace=True)

            # Transform location data into latitude and longitude features
            data['LATITUDE'] = data['Location'].str.extract(r'\(([-+]?[0-9]*\.?[0-9]+),\s*([-+]?[0-9]*\.?[0-9]+)\)')[0].astype(float)
            data['LONGITUDE'] = data['Location'].str.extract(r'\(([-+]?[0-9]*\.?[0-9]+),\s*([-+]?[0-9]*\.?[0-9]+)\)')[1].astype(float)

            # Drop the original location column
            data.drop(columns=['Location', 'Lat', 'Long'], axis=1, inplace=True)

            # Split the REPORTING_AREA feature into numeric and non-numeric components
            data['REPORTING_AREA'] = data['REPORTING_AREA'].str.extract(r'(\d+)')[0].astype(float)
            data['REPORTING_AREA_STR'] = data['REPORTING_AREA'].astype(str)
            data.drop('REPORTING_AREA', axis=1, inplace=True)

            # Drop OFFENSE_DESCRIPTION column
            data.drop(columns=['OFFENSE_DESCRIPTION'], axis=1, inplace=True)

            logging.info("Calling get tranformation function")

            preprocessor_obj = self.get_data_transformer_object()

            # split target variable and training variable
            target_feature = data['OFFENSE_CODE_GROUP']
            data = data.drop(columns=['OFFENSE_CODE_GROUP'], axis=1)

            data_arr = preprocessor_obj.fit_transform(data)
            label_encoder = LabelEncoder()
            target_arr = label_encoder.fit_transform(target_feature)

            logging.info("Saving preprocessor object")

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path, 
                preprocessor_obj
                )


            logging.info("Saved preprocessor object")

            logging.info("Data transformation is completed")

            return data_arr, target_arr

        except Exception as e:
            raise CustomException(e,sys)
