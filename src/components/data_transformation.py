import os, sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    pre_processor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tansformation_config = DataTransformationConfig()

    def get_data_tranformation_obj(self):
        try:
            numerical_features= ['age','bmi','children']
            categorical_features=['sex', 'smoker', 'region']
            

            num_pipeline=Pipeline(steps=[
                ('scaler', StandardScaler(with_mean=False))
            ])

            cat_pipeline=Pipeline(steps=[
                ('encoding', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info('Numerical columns scaling is completed')
            logging.info('Categorical columns encoding and scaling is completed')

            preprocessor=ColumnTransformer([
                ('num_transformation', num_pipeline, numerical_features),
                ('cat_transformation', cat_pipeline, categorical_features)])
            
            logging.info('Data transformer object is ready')
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test datasets')

            logging.info('Getting preprocessor object')
            preprocessor_obj = self.get_data_tranformation_obj()
            target_feature= 'charges'

            train_set = train_df.drop([target_feature], axis=1)
            test_set = test_df.drop([target_feature], axis=1)

            train_target = train_df[target_feature]
            test_target = test_df[target_feature]
            logging.info('Applying preprocessing object on train and test data')

            train_set_arr = preprocessor_obj.fit_transform(train_set)
            test_set_arr = preprocessor_obj.transform(test_set)

            train_array = np.c_[train_set_arr,np.asarray( train_target)]
            test_array = np.c_[test_set_arr, np.asarray(test_target)]

            logging.info('Saved preprocessing objects')

            save_object(
                file_path = self.data_tansformation_config.pre_processor_obj_file_path,
                object = preprocessor_obj
            )

            return (train_array, test_array, self.data_tansformation_config.pre_processor_obj_file_path)



        except Exception as e:
            raise CustomException(e,sys)

        

                