"""
In this script we can train our model.
We can train different model and select the best model
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor,GradientBoostingRegressor,
                                BaggingRegressor,AdaBoostRegressor
                            )
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import optuna
import mlflow
from src.utils import train_models,save_file
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
            # "xgboost":XGBRegressor()
        }
        
        # model hyperparameter
       
        # Train the models
        results=train_models(model_dic=model_dic,x_train=x_train,x_test=x_test,
                                        y_train=y_train,y_test=y_test)

        best_model_name=results.sort_values(by='Adjusted_R2',ascending=False)['model_name'].values[0]
        

        best_model=model_dic[best_model_name]
        logging.info(f"Best model name: {best_model_name}")
        # hyperparamter using optuna
        logging.info("Hyperparameter tunning")
        best_params=self.optimize_hyperparameters(model_name=best_model_name,x_train=x_train,y_train=y_train,
                                                    x_test=x_test,y_test=y_test)
        
        # train the best model with best parameters
        if best_model_name == "RandomForest":
            best_model = RandomForestRegressor(**best_params)
        elif best_model_name == "Gradient_Boosting":
            best_model = GradientBoostingRegressor(**best_params)
        elif best_model_name == "Bagging":
            best_model = BaggingRegressor(**best_params)
        elif best_model_name == "AdaBoost":
            best_model = AdaBoostRegressor(**best_params)
        elif best_model_name == "DecessionTree":
            best_model = DecisionTreeRegressor(**best_params)
        else:
            best_model = LinearRegression()

        mlflow.autolog()
        with mlflow.start_run(run_name="Hyper_Optus", nested=True) as run:
            best_model.fit(x_train,y_train)
            mlflow.log_params(best_model.get_params())

            y_pred = best_model.predict(x_test)

            final_mae=mean_absolute_error(y_test,y_pred)
            final_mse=mean_squared_error(y_test,y_pred)
            final_r2 = r2_score(y_test, y_pred)
            final_adjusted_r2 = 1 - ((1 - final_r2) * (x_train.shape[0] - 1)) / (x_train.shape[0] - x_train.shape[1] - 1)

            logging.info(f"Final R2 score after hyperparameter tuning: {final_r2}")
            logging.info(f"Final adjusted R2 score after hyperparameter tuning: {final_adjusted_r2}")


            # log final metrics
            mlflow.log_metric("final_mae", final_mae)
            mlflow.log_metric("final_mse", final_mse)
            mlflow.log_metric("final_r2_score", final_r2)
            mlflow.log_metric("final_adjusted_r2", final_adjusted_r2)

            # log final model
            mlflow.sklearn.log_model(best_model, "final_model")

            
        return best_model, best_params
        

    def objective(self, trial, model_name, x_train, y_train, x_test, y_test):
        if model_name == "RandomForest":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        elif model_name == "Gradient_Boosting":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.1)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        elif model_name == "Bagging":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            max_samples = trial.suggest_float("max_samples", 0.5, 1.0)
            max_features = trial.suggest_float("max_features", 0.5, 1.0)
            model = BaggingRegressor(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features)
        elif model_name == "AdaBoost":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)
            learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 1.0)
            model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
        elif model_name == "DecessionTree":
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
        elif model_name == "XGBoost":
            params = {
            'booster': trial.suggest_categorical("booster", ['gbtree', 'dart']),
            'eta': trial.suggest_loguniform("eta", 0.01, 0.3),
            'max_depth': trial.suggest_int("max_depth", 3, 20),
            'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
            'subsample': trial.suggest_float("subsample", 0.6, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
            'alpha': trial.suggest_loguniform("alpha", 0.1, 10),
            'gamma': trial.suggest_float("gamma", 0, 5),
            'n_estimators': trial.suggest_int("n_estimators", 10, 200)
        }
            model = XGBRegressor(
                booster=params['booster'],
                eta=params['eta'],
                max_depth=params['max_depth'],
                min_child_weight=params['min_child_weight'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                alpha=params['alpha'],
                gamma=params['gamma'],
                n_estimators=params['n_estimators']
            )
        else:
            model = LinearRegression()
        
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        adjusted_r2 = 1 - ((1 - r2) * (x_train.shape[0] - 1)) / (x_train.shape[0] - x_train.shape[1] - 1)
        return adjusted_r2

    def optimize_hyperparameters(self, model_name, x_train, y_train, x_test, y_test):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, model_name, x_train, y_train, x_test, y_test), n_trials=30)
        
        best_params = study.best_trial.params
        logging.info(f"Best hyperparameters for {model_name}: {best_params}")
        return best_params

