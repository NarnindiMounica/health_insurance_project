import os, sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exceptions import CustomException

def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(object, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(x_train, y_train, x_test, y_test, models, parameters):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            param = list(parameters.keys())[i]

            grid_search = GridSearchCV(model, param_grid = parameters[param],cv=5)
            grid_search.fit(x_train, y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(x_train, y_train)

            train_predict = model.predict(x_train)
            test_predict = model.predict(x_test)
            train_score = r2_score(y_train, train_predict)
            test_score = r2_score(y_test, test_predict)

            report[list(models.keys())[i]] = test_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
