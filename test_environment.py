"""
This is test environment in this file we can test our pipeline.
Pipeline like data ingestion transformaton and model building pieplines etc
"""

from src.Components.data_ingestion import inisiate_data_ingestion
from src.Components.feature_engnering import feature_engnering_config
from src.Components.data_transformation import data_transformation_config
from src.Components.model_training import Model_Training_Config
from src.utils import save_file
import os

if __name__=="__main__":
    # Data Ingestion
    df=inisiate_data_ingestion()
    
    # Feature engnering
    fe=feature_engnering_config(df=df)
    train_data_path,test_data_path=fe.inisiate_feature_engnering()

    # Data transformation
    dt=data_transformation_config(train_data_path=train_data_path,
                                    test_data_path=test_data_path)
    train_array,test_array=dt.inisiate_data_transformation()

    # Model training
    trainer=Model_Training_Config(train_array=train_array,test_array=test_array)
    best_model,_=trainer.train()

    # Save the best model
    best_model_path = os.path.join("Models", "best_model.pkl")
    save_file(obj=best_model, file_path= best_model_path)