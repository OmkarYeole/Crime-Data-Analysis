import pandas as pd
import numpy as np
import sys
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging

class DataTransformation:
    def __init__(self, train_path:str):
        self.train_path = train_path
        
    def initiate_data_transformation(self):
        logging.info("Entered the data transformation method or component")
        try:
            # Read in the training data
            train_data = pd.read_csv(self.train_path, parse_dates=['OCCURRED_ON_DATE'], low_memory=False)

            # drop INCIDENT_NUMBER feature
            train_data.drop(columns=['INCIDENT_NUMBER'], axis=1, inplace=True)

            # Imputing SHOOTING columns as it contains more NAN values
            train_data['SHOOTING'] = np.where(train_data['SHOOTING'] == 'Y', 1, 0)

            #  Encode categorical variables as integers
            label_encoder = LabelEncoder()
            train_data['DAY_OF_WEEK'] = label_encoder.fit_transform(train_data['DAY_OF_WEEK'])
            train_data['OFFENSE_CODE_GROUP'] = label_encoder.fit_transform(train_data['OFFENSE_CODE_GROUP'])
            train_data['DISTRICT'] = label_encoder.fit_transform(train_data['DISTRICT'])

            # Extract date feature
            train_data['OCCURRED_ON_DATE'] = pd.to_datetime(train_data['OCCURRED_ON_DATE'])
            train_data['YEAR'] = train_data['OCCURRED_ON_DATE'].dt.year
            train_data['MONTH'] = train_data['OCCURRED_ON_DATE'].dt.month
            train_data['DAY'] = train_data['OCCURRED_ON_DATE'].dt.day
            train_data['HOUR'] = train_data['OCCURRED_ON_DATE'].dt.hour

            # Drop the original date column
            train_data.drop('OCCURRED_ON_DATE', axis=1, inplace=True)

            # Transform location data into latitude and longitude features
            train_data['LATITUDE'] = train_data['Location'].str.extract(r'\(([-+]?[0-9]*\.?[0-9]+),\s*([-+]?[0-9]*\.?[0-9]+)\)')[0].astype(float)
            train_data['LONGITUDE'] = train_data['Location'].str.extract(r'\(([-+]?[0-9]*\.?[0-9]+),\s*([-+]?[0-9]*\.?[0-9]+)\)')[1].astype(float)

            # Drop the original location column
            train_data.drop(columns=['Location', 'Lat', 'Long'], axis=1, inplace=True)

            # Split the REPORTING_AREA feature into numeric and non-numeric components
            train_data['REPORTING_AREA'] = train_data['REPORTING_AREA'].str.extract(r'(\d+)')[0].astype(float)
            train_data['REPORTING_AREA_STR'] = train_data['REPORTING_AREA'].astype(str)
            train_data.drop('REPORTING_AREA', axis=1, inplace=True)

            # Encode categorical variables as integers
            label_encoder = LabelEncoder()
            train_data['OFFENSE_CODE_GROUP'] = label_encoder.fit_transform(train_data['OFFENSE_CODE_GROUP'])
            train_data['UCR_PART'] = label_encoder.fit_transform(train_data['UCR_PART'])
            train_data['STREET'] = label_encoder.fit_transform(train_data['STREET'])
            train_data['REPORTING_AREA_STR'] = label_encoder.fit_transform(train_data['REPORTING_AREA_STR'])

            # Drop OFFENSE_DESCRIPTION column
            train_data.drop(columns=['OFFENSE_DESCRIPTION'], axis=1, inplace=True)

            logging.info("Data transformation is completed")

            return train_data

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