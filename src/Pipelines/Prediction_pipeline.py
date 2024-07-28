import os
import mlflow
import mlflow.pyfunc
import pickle as pkl
import logging
logging.basicConfig(level=logging.INFO)

# load the model form mlflow reggistry
# model_name = "flight_price_prediction_model"
# model_version = 1

# # Load the model from the MLflow model registry
# model_uri = f"models:/{model_name}/{model_version}"
# model = mlflow.pyfunc.load_model(model_uri)
# print("model successfully loaded")

def load_model(model_name,model_version):
    """
    This fun is responsible for loading the model form mlflow registry.
    After loading the model it can return the model.
    """
    try:
        logging.info("Loading the model")
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
        print("model successfully loaded")
    except Exception as e:
        print("Model not found")


# load the porcessor
def process_input(data):
    """
    This fun is responsible for preporcess the input and convert them into correct format so that model can understand.
    """
    try:
        logging.info("Loading the processor")
        with open('Models/processor.pkl',"rb")as f:
            processor=pkl.load(f)
        data=processor.transform(data)
        return data
    except Exception as e:
        return e


# Predict fun
def predict(data):
    """
    This fun is responsible to predict the value.
    """
    # load the process_input to process the data
    process_data=process_input(data=data)

    # load the model
    model=load_model(model_name="flight_price_prediction_model",model_version=1)
    y_pred=model.predict(process_data)

    return y_pred[0]