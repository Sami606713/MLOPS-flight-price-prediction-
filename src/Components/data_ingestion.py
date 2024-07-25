import pandas as pd
from src.utils import read_data,clean_data
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)


def inisiate_data_ingestion():
    """
    In this fun first we will read the data.
    Then we will apply basic preprocessing like handling missing valeues removing duplicates and handling datatypes
    We can save the data we will get in the raw folder
    """
    raw_data_path=os.path.join("Data/Raw","raw.csv")

    logging.info("Getting the data")
    df=read_data()

    logging.info("Apply data cleaning")

    df=clean_data(df=df)

    logging.info("saving the raw data")
    df.to_csv(raw_data_path,index=False)

    return df

# if __name__=="__main__":
#     inisiate_data_ingestion()