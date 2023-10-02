import os, sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    model_train_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:   
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 

    def initiate_model_training(self,train_array, test_array):
        try:
            logging.info('Splitting training and testing data into x and y')
            x_train, y_train, x_test, y_test = (train_array[:,:-1],  train_array[:,-1], test_array[:,:-1], test_array[:,-1])

            models= {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "K Neighbors": KNeighborsRegressor(),
                "Ada Boost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Cat Boosting": CatBoostRegressor(verbose=False),
                "XGBoost": XGBRegressor()
            }

            model_report:dict = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)

            logging.info('Getting best model from evaluated models')
            best_model_name = max(zip(model_report.items()))[0][0]
            best_model_score = max(zip(model_report.items()))[0][1]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            logging.info('Best model found on training and testing data')

            save_object(
                file_path = self.model_trainer_config.model_train_path,
                object = best_model
            )

            best_pred = best_model.predict(x_test)
            best_r2_score = r2_score(y_test, best_pred)

            return best_r2_score

        except Exception as e:
            raise CustomException(e,sys) 



