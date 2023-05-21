import os
import sys

from dataclasses import dataclass
from sklearn.ensemble import (

    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:

    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self, train_array,test_array):

        try:

            X_train, y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            
            )
            # print(train_array)
            models = {
                'Adaboost Regressor' : AdaBoostRegressor(),
                'Gradient Boosting' : GradientBoostingRegressor(),
                'Randon forest':RandomForestRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors classifier':KNeighborsRegressor(),
                'Descision Tree' : DecisionTreeRegressor(),
                'XGBClassifier':XGBRegressor()

            }

            model_reports: dict=evaluate_model(X_train,y_train,X_test,y_test, models)

            best_model = max(zip(model_reports.values(), model_reports.keys()))[1]  
            
            if model_reports[best_model] < 0.6:
                raise CustomException('No model is best suited ')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_hat = models[best_model].predict(X_test)
            r2 = r2_score(y_test, y_hat)
            return r2
        except Exception as e:
            raise CustomException(e,sys)
        