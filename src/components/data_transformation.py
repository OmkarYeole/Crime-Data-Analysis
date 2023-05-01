import pandas as pd
import numpy as np
import sys
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging

class DataTransformation:
    def __init__(self, data_path:str):
        self.data_path = data_path
        
    def initiate_data_transformation(self):
        logging.info("Entered the data transformation method or component")
        try:
            # Read the data
            data = pd.read_csv(self.data_path, parse_dates=['OCCURRED_ON_DATE'], low_memory=False)

            # drop INCIDENT_NUMBER feature
            data.drop(columns=['INCIDENT_NUMBER'], axis=1, inplace=True)

            # Imputing SHOOTING columns as it contains more NAN values
            data['SHOOTING'] = np.where(data['SHOOTING'] == 'Y', 1, 0)

            #  Encode categorical variables as integers
            label_encoder = LabelEncoder()
            data['DAY_OF_WEEK'] = label_encoder.fit_transform(data['DAY_OF_WEEK'])
            data['OFFENSE_CODE_GROUP'] = label_encoder.fit_transform(data['OFFENSE_CODE_GROUP'])
            data['DISTRICT'] = label_encoder.fit_transform(data['DISTRICT'])

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

            # Encode categorical variables as integers
            label_encoder = LabelEncoder()
            data['OFFENSE_CODE_GROUP'] = label_encoder.fit_transform(data['OFFENSE_CODE_GROUP'])
            data['UCR_PART'] = label_encoder.fit_transform(data['UCR_PART'])
            data['STREET'] = label_encoder.fit_transform(data['STREET'])
            data['REPORTING_AREA_STR'] = label_encoder.fit_transform(data['REPORTING_AREA_STR'])

            # Drop OFFENSE_DESCRIPTION column
            data.drop(columns=['OFFENSE_DESCRIPTION'], axis=1, inplace=True)

            logging.info("Data transformation is completed")

            return data

        except Exception as e:
            raise CustomException(e,sys)
            

if __name__ == "__main__":
    ######
    # After this the train and test path
    # will return by data_ingestion.py file
    #####
    # Define the paths to the training and testing data
    train_path = 'artifacts/train.csv'

    # Instantiate the DataTransformation class and initiate data transformation
    dt = DataTransformation(train_path)
    train_data_transformed = dt.initiate_data_transformation()

    # Print the transformed data for inspection
    print(train_data_transformed.head())