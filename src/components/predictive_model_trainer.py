import os
import sys
import numpy as np
from dataclasses import dataclass


from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")

            X_train,y_train,X_test,y_test=(
                np.delete(train_array , 1, axis=1),
                train_array[:,1],
                np.delete(test_array , 1, axis=1),
                test_array[:,1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            params={
                "Decision Tree": {
                    'criterion':['gini', 'entropy'],
                    'max_depth':[None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256],
                    'criterion':['gini', 'entropy'],
                    'max_depth':[None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256],
                    'max_depth':[None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                "Logistic Regression":{
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                "XGBClassifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256],
                    'max_depth':[None, 5, 10, 15],
                    'min_child_weight': [1, 5, 10],
                },
                "CatBoosting Classifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.1,1,10,100],
                    'iterations': [30, 50, 100],
                    'cat_features': [0, 1, 2],
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256],
                    'algorithm': ['SAMME', 'SAMME.R']
                }

            }

            '''

            for easy porpose
            params={
                "Decision Tree": {
                'criterion':['gini', 'entropy']
                },
                "Random Forest":{
                'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                'learning_rate':[.1,.01,.05,.001]
                },
                "Logistic Regression":{
                'C': [0.01, 0.1, 1, 10]
                },
                "XGBClassifier":{
                'learning_rate':[.1,.01,.05,.001],
                },
                "CatBoosting Classifier":{
                'depth': [6,8,10]
                },
                "AdaBoost Classifier":{
                'learning_rate':[.1,.01,0.5,.001]
                }   
            }

            '''

            model_report:dict=evaluate_models(

                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test,
                models=models,
                param=params,
                scoring_function=f1_score
                
                )
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":

    train_data = pd.read_csv()