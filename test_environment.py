"""
This is test environment in this file we can test our pipeline.
Pipeline like data ingestion transformaton and model building pieplines etc
"""

from src.Components.data_ingestion import inisiate_data_ingestion
from src.Components.feature_engnering import feature_engnering_config

if __name__=="__main__":
    # Data Ingestion
    df=inisiate_data_ingestion()
    
    # Feature engnering
    fe=feature_engnering_config(df=df)
    train_data_path,test_data_path=fe.inisiate_feature_engnering()