"""
This is test environment in this file we can test our pipeline.
Pipeline like data ingestion transformaton and model building pieplines etc
"""

from src.Components.data_ingestion import inisiate_data_ingestion

if __name__=="__main__":
    # Data Ingestion
    df=inisiate_data_ingestion()