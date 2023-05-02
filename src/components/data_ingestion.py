import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# import the exception module from the correct relative path
from src.exception import CustomException
# import the logger module from the correct relative path
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.predictive_model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            ####
            # Change the below code so that
            # it take it from cloud
            ####
            
            with open('notebook/crime.csv', 'rb') as file:
                df = pd.read_csv(file, encoding='iso-8859-1')
            logging.info('Read the dataset as dataframe')


            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":

    data_ingestion_obj = DataIngestion()
    train_path, test_path = data_ingestion_obj.initiate_data_ingestion()

    # processing through data tranformation
    data_tranformation_obj = DataTransformation()

    # passing the tarining and testing dataset
    train_arr = data_tranformation_obj.initiate_data_transformation(train_path)
    test_arr = data_tranformation_obj.initiate_data_transformation(test_path)

    # pass the dataset into model trainer file
    model_trainer_obj = ModelTrainer()
    print(model_trainer_obj.initiate_model_trainer(train_arr,test_arr))


