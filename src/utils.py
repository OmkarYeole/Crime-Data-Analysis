import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    logging.info("Entered into save object function in utils.py")
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Succesfully saved the {obj} at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param, scoring_function):
    logging.info("Entered into evauate models function in util.py file")
    try:

        scoring = make_scorer(scoring_function, average='weighted')
        report = {}

        for name, model in models.items():
            gs = GridSearchCV(model, param[name], cv=3,scoring=scoring, n_jobs=-1, verbose=3, return_train_score=True, refit=True, error_score='raise', pre_dispatch='2*n_jobs')
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            score = scoring_function(y_test, y_pred)
            report[name] = score

            logging.info("Successfully completed evauate models function in util.py file")

        return report
    
    except Exception as e:
        raise CustomException(e,sys)