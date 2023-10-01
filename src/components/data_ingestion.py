import os, sys
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered data ingestion phase')
        try:
            df = pd.read_csv('D:\\ML Projects\\Health_Insurance_Project\\Notebook\\data\\health_insurance.csv')
            logging.info('Read data into dataset')

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Initiate train-test split')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=1)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of the data is completed')


            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)       

if __name__=='__main__':
    ingestion_obj = DataIngestion()
    train_path, test_path=ingestion_obj.initiate_data_ingestion()  
     
    transformation_obj = DataTransformation()
    transformation_obj.initiate_data_transformation(train_path, test_path)     



            