"""
In this script we can train our model.
We can train different model and select the best model
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor,GradientBoostingRegressor,
                                BaggingRegressor,AdaBoostRegressor
                            )
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import train_models
import logging
logging.basicConfig(level=logging.INFO)


class Model_Training_Config:
    def __init__(self,train_array,test_array):
        self.train_array=train_array
        self.test_array=test_array
    
    def train(self):
        # saperate the input and output col
        logging.info("saperate the input and output col")
        x_train=self.train_array[:,:-1]
        y_train=self.train_array[:,-1]

        x_test=self.test_array[:,:-1]
        y_test=self.test_array[:,-1]
        logging.info("Saperate input and output done")
        # print(y_train,y_test)

        # Model dic
        model_dic={
            "LinearRegression":LinearRegression(),
            "RandomForest":RandomForestRegressor(),
            "AdaBoost":AdaBoostRegressor(),
            "Gradient_Boosting":GradientBoostingRegressor(),
            "Bagging":BaggingRegressor(),
            "DecessionTree":DecisionTreeRegressor(),
            "xgboost":XGBRegressor()
        }
        
        # model hyperparameter

        # Train the models
        results=train_models(model_dic=model_dic,x_train=x_train,x_test=x_test,
                                        y_train=y_train,y_test=y_test)

        best_model_name=results.sort_values(by='Adjusted_R2',ascending=False)['model_name'].values[0]
        # print(best_model_name)

        best_model=model_dic[best_model_name]
        print("Best model name: ",best_model_name)
        # print("best model: ",best_model)
