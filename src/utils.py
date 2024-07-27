import os
import pandas as pd
import numpy as np
import logging 
import pickle as pkl
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)

# Make a fun for reading data
def read_data():
    """
    This fun is responsible for reading the data form any source.
    
    It can read the data and return the data as a dataframe.
    """

    # read the data
    logging.info("Reading the data from source")
    df=pd.read_csv("Given_data/flight_price.csv")

    logging.info("Successfully read the data")

    return df


def clean_data(df):
    """
    This function is responsible for cleaning the data.
    Cleaning data like correct data types handling missing values drop duplicates etc.
    After Analysis we can see that there is very less nbr of missing values so we can drop missing values.
    We can also see that data contain duplicates so we can drop the duplicates.
    """

    # Remove missing values
    logging.info("Removing Null Values")
    df.dropna(inplace=True)

    # Remove Duplicates
    logging.info("Removing Duplicates Values")
    df.drop_duplicates(inplace=True)

    # Change the data type of data of journey b/c it is in object form
    df['Date_of_Journey']=pd.to_datetime(df['Date_of_Journey'],format='%d/%m/%Y')

    logging.info("Data Cleaning Successfully")

    return df


def save_file(obj,file_path):
    """
    This fun is responsible for save the binary file like processor,models etc
    """
    with open(file_path,"wb") as f:
        pkl.dump(obj,f)


def evulation(model_name,actual,y_pred,n,k):
    """
    This fun is responsible for evulate the performace of model
    """
    r2=r2_score(actual,y_pred)
    return {
        "model_name":model_name,
        "Mean_Squared_Error":mean_squared_error(actual,y_pred),
        "Mean_absolute_error":mean_absolute_error(actual,y_pred),
        "R2_Score":r2,
        "Adjusted_r2":1 - ((1 - r2) * (n - 1)) / (n - k - 1)
    }


def train_models(model_dic,x_train,y_train,x_test,y_test):
    mlflow.autolog()
    evulation={
        "model_name":[],
        "MAE":[],
        "MSE":[],
        "R2":[],
        "Adjusted_R2":[],
        "train_score":[],
        "test_score":[]
    }
    for model_name,model in model_dic.items():
        with mlflow.start_run(run_name=model_name) as run:
            logging.info(f"Training Model {model_name}")
            model.fit(x_train,y_train)
            mlflow.log_params(model.get_params())

            # Prediction
            logging.info(f"{model_name} prediction")
            y_pred=model.predict(x_test)

            # Evulation
            logging.info(f"{model_name} evulation")
            train_score=cross_val_score(model,x_train,y_train,cv=5,scoring="r2")
            test_score=cross_val_score(model,x_test,y_test,cv=5,scoring='r2')


            mae=mean_absolute_error(y_test,y_pred)
            mse=mean_squared_error(y_test,y_pred)
            r2=r2_score(y_test,y_pred)
            adjusted=1 - ((1 - r2) * (x_train.shape[0] - 1)) / (x_train.shape[0] - x_train.shape[1] - 1)

            evulation['model_name'].append(model_name)
            evulation['MAE'].append(mae)
            evulation['MSE'].append(mse)
            evulation['R2'].append(r2)
            evulation['Adjusted_R2'].append(adjusted)
            evulation['train_score'].append(train_score.mean())
            evulation['test_score'].append(test_score.mean())  
        
            # Log the matrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("adjusted_r2", adjusted)
            mlflow.log_metric("train_score", train_score.mean())
            mlflow.log_metric("test_score", test_score.mean())


            # log the model
            mlflow.sklearn.log_model(model, model_name)
    return pd.DataFrame(evulation)
